from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
from efficient_sam.utils.boundary_utils import compute_sobel_edges_from_labels
from efficient_sam.utils.semantic_label_utils import (
    decode_cityscapes_like_label_to_train_ids,
    encode_train_ids_to_color,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _normalize_name(stem: str) -> str:
    suffixes = [
        "_leftImg8bit",
        "_gtFine_labelIds",
        "_gtFine_labelTrainIds",
        "_labelIds",
        "_labelTrainIds",
    ]
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _make_relative_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    parts[-1] = _normalize_name(parts[-1])
    return "/".join(parts)


class ValInferenceDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        input_size: int,
        label_assume_bgr: bool,
        label_max_color_distance: float,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.label_assume_bgr = label_assume_bgr
        self.label_max_color_distance = label_max_color_distance
        self.samples = self._pair_samples()
        if not self.samples:
            raise RuntimeError(
                f"No matched image/label pairs found in {self.image_dir} and {self.label_dir}"
            )

    def _collect_files(self, root: Path) -> List[Path]:
        return sorted(
            [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )

    def _pair_samples(self) -> List[Tuple[Path, Path, str]]:
        image_files = self._collect_files(self.image_dir)
        label_files = self._collect_files(self.label_dir)
        label_map = {_make_relative_key(path, self.label_dir): path for path in label_files}
        matched: List[Tuple[Path, Path, str]] = []
        for image_path in image_files:
            rel_key = _make_relative_key(image_path, self.image_dir)
            label_path = label_map.get(rel_key)
            if label_path is not None:
                matched.append((image_path, label_path, rel_key))
        return matched

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _full_image_bbox_prompt(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points = torch.tensor(
            [[[0.0, 0.0], [float(width - 1), float(height - 1)]]], dtype=torch.float32
        )
        point_labels = torch.tensor([[2, 3]], dtype=torch.int64)
        return points, point_labels

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        image_path, label_path, rel_key = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        if self.input_size > 0:
            image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
            label = label.resize((self.input_size, self.input_size), resample=Image.NEAREST)

        image_tensor = pil_to_tensor(image)
        label_np = np.array(label, dtype=np.uint8)
        train_ids = decode_cityscapes_like_label_to_train_ids(
            label_np,
            assume_bgr=self.label_assume_bgr,
            max_color_distance=self.label_max_color_distance,
        )
        label_tensor = torch.from_numpy(train_ids.astype(np.int64))
        points, point_labels = self._full_image_bbox_prompt(
            height=label_tensor.shape[0], width=label_tensor.shape[1]
        )
        return image_tensor, label_tensor, points, point_labels, rel_key


def compute_semantic_iou_and_dice(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    iou_list = []
    dice_list = []
    valid = gt_labels != ignore_index
    for cls in range(num_classes):
        pred_c = (pred_labels == cls) & valid
        gt_c = (gt_labels == cls) & valid
        tp = (pred_c & gt_c).sum().float()
        fp = (pred_c & (~gt_c)).sum().float()
        fn = ((~pred_c) & gt_c).sum().float()
        denom_iou = tp + fp + fn
        denom_dice = 2.0 * tp + fp + fn
        if denom_iou > 0:
            iou_list.append((tp + eps) / (denom_iou + eps))
        if denom_dice > 0:
            dice_list.append((2.0 * tp + eps) / (denom_dice + eps))
    if not iou_list:
        zero = torch.tensor(0.0, device=pred_labels.device)
        return zero, zero
    return torch.stack(iou_list).mean(), torch.stack(dice_list).mean()


def compute_pixel_accuracy(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = gt_labels != ignore_index
    correct = (pred_labels == gt_labels) & valid
    return (correct.sum().float() + eps) / (valid.sum().float() + eps)


def compute_boundary_f1_per_sample(
    pred_boundary: torch.Tensor, gt_boundary: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred = (pred_boundary > 0.5).float()
    gt = (gt_boundary > 0.5).float()
    true_positive = (pred * gt).sum(dim=(1, 2, 3))
    precision = (true_positive + eps) / (pred.sum(dim=(1, 2, 3)) + eps)
    recall = (true_positive + eps) / (gt.sum(dim=(1, 2, 3)) + eps)
    return (2.0 * precision * recall + eps) / (precision + recall + eps)


def build_model(model_variant: str, num_classes: int):
    if model_variant.lower() == "vits":
        return build_efficient_sam_vits(
            enable_boundary_decoder=True,
            semantic_num_classes=num_classes,
            enable_semantic_head=True,
        )
    return build_efficient_sam_vitt(
        enable_boundary_decoder=True,
        semantic_num_classes=num_classes,
        enable_semantic_head=True,
    )


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    default_epoch_ckpt = Path("checkpoints/checkpoint_epoch_30.pth")
    if default_epoch_ckpt.exists():
        return default_epoch_ckpt
    return Path("checkpoints/best_model.pth")


def save_trainid_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def save_color_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic val inference with boundary-aware EfficientSAM."
    )
    parser.add_argument("--dataset-root", type=str, default="/home/raid/vishal10/dataset")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--image-dir-name", type=str, default="img")
    parser.add_argument("--label-dir-name", type=str, default="label")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-variant", type=str, default="vitt")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="inference_outputs/val_semantic")
    parser.add_argument("--label-assume-bgr", action="store_true", default=True)
    parser.add_argument("--label-assume-rgb", action="store_false", dest="label_assume_bgr")
    parser.add_argument("--label-max-color-distance", type=float, default=55.0)
    parser.add_argument("--sobel-threshold", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / args.val_split / args.image_dir_name
    label_dir = dataset_root / args.val_split / args.label_dir_name
    dataset = ValInferenceDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        input_size=args.input_size,
        label_assume_bgr=args.label_assume_bgr,
        label_max_color_distance=args.label_max_color_distance,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model_variant, num_classes=args.num_classes).to(device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    output_dir = Path(args.output_dir)
    pred_trainid_dir = output_dir / "pred_trainids"
    pred_color_dir = output_dir / "pred_color"
    gt_trainid_dir = output_dir / "gt_trainids"
    gt_color_dir = output_dir / "gt_color"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_iou = 0.0
    total_dice = 0.0
    total_boundary_f1 = 0.0
    total_pixacc = 0.0
    num_samples = 0

    progress = tqdm(dataloader, desc="Inference")
    for images, gt_labels, points, point_labels, rel_keys in progress:
        images = images.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            pred_masks, _, pred_boundaries, pred_semantic_logits = model.forward_with_boundary_and_semantics(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
                multimask_output=False,
            )
            del pred_masks
            pred_boundary = pred_boundaries[:, 0, 0, :, :].unsqueeze(1)
            semantic_logits = pred_semantic_logits[:, 0, :, :, :]

        pred_labels = torch.argmax(semantic_logits, dim=1)
        gt_boundary = compute_sobel_edges_from_labels(
            gt_labels,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            threshold=args.sobel_threshold,
        )
        pred_boundary_bin = (pred_boundary > 0.5).float()

        iou, dice = compute_semantic_iou_and_dice(
            pred_labels, gt_labels, num_classes=args.num_classes, ignore_index=args.ignore_index
        )
        pixacc = compute_pixel_accuracy(pred_labels, gt_labels, ignore_index=args.ignore_index)
        boundary_f1 = compute_boundary_f1_per_sample(pred_boundary_bin, gt_boundary).mean()

        batch_size = images.shape[0]
        total_iou += float(iou.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_boundary_f1 += float(boundary_f1.item()) * batch_size
        total_pixacc += float(pixacc.item()) * batch_size
        num_samples += batch_size

        pred_np = pred_labels.detach().cpu().numpy().astype(np.uint8)
        gt_np = gt_labels.detach().cpu().numpy().astype(np.uint8)

        for idx in range(batch_size):
            rel_path = Path(rel_keys[idx]).with_suffix(".png")
            pred_trainid = pred_np[idx]
            gt_trainid = gt_np[idx]
            pred_color = encode_train_ids_to_color(pred_trainid, bgr_output=True)
            gt_color = encode_train_ids_to_color(gt_trainid, bgr_output=True)
            save_trainid_mask(pred_trainid, pred_trainid_dir / rel_path)
            save_trainid_mask(gt_trainid, gt_trainid_dir / rel_path)
            save_color_mask(pred_color, pred_color_dir / rel_path)
            save_color_mask(gt_color, gt_color_dir / rel_path)

        progress.set_postfix(
            miou=f"{(total_iou / num_samples):.4f}",
            mdice=f"{(total_dice / num_samples):.4f}",
            pixacc=f"{(total_pixacc / num_samples):.4f}",
            boundary_f1=f"{(total_boundary_f1 / num_samples):.4f}",
        )

    summary: Dict[str, float | int | str] = {
        "checkpoint": str(checkpoint_path),
        "num_samples": num_samples,
        "mean_iou": total_iou / max(num_samples, 1),
        "mean_dice": total_dice / max(num_samples, 1),
        "pixel_accuracy": total_pixacc / max(num_samples, 1),
        "mean_boundary_f1": total_boundary_f1 / max(num_samples, 1),
    }

    summary_path = output_dir / "stats.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    print("Inference complete.")
    print(json.dumps(summary, indent=2))
    print(f"Saved predicted trainId masks to: {pred_trainid_dir}")
    print(f"Saved predicted color masks to: {pred_color_dir}")
    print(f"Saved GT trainId masks to: {gt_trainid_dir}")
    print(f"Saved GT color masks to: {gt_color_dir}")
    print(f"Saved stats to: {summary_path}")


if __name__ == "__main__":
    main()
