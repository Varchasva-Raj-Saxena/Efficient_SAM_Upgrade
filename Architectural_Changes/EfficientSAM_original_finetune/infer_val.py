from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cityscapes_semantic_utils import (
    boundary_f1_score,
    build_prompt_targets_from_train_ids,
    compute_pixel_accuracy,
    compute_semantic_iou_and_dice,
    compute_sobel_edges_from_labels,
    decode_cityscapes_like_label_to_train_ids,
    encode_train_ids_to_color,
    limit_samples,
    pair_image_label_files,
)
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CityscapesValPromptDataset(Dataset):
    def __init__(
        self,
        *,
        image_dir: Path,
        label_dir: Path,
        input_size: int,
        num_classes: int,
        ignore_index: int,
        max_queries_per_image: int,
        min_class_pixels: int,
        label_assume_bgr: bool,
        label_max_color_distance: float,
        max_samples: int,
        seed: int,
    ) -> None:
        super().__init__()
        samples = pair_image_label_files(image_dir=image_dir, label_dir=label_dir)
        self.samples = limit_samples(samples, max_samples=max_samples)
        if not self.samples:
            raise RuntimeError(f"No matched pairs in {image_dir} and {label_dir}")

        self.input_size = input_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.max_queries_per_image = max_queries_per_image
        self.min_class_pixels = min_class_pixels
        self.label_assume_bgr = label_assume_bgr
        self.label_max_color_distance = label_max_color_distance
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_path, rel_key = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.input_size > 0:
            image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
            label = label.resize((self.input_size, self.input_size), resample=Image.NEAREST)

        image_tensor = pil_to_tensor(image)
        label_rgb = np.asarray(label, dtype=np.uint8)
        train_ids = decode_cityscapes_like_label_to_train_ids(
            label_rgb,
            assume_bgr=self.label_assume_bgr,
            max_color_distance=self.label_max_color_distance,
        )
        points, point_labels, _target_masks, _valid_masks, class_ids = build_prompt_targets_from_train_ids(
            train_ids=train_ids,
            max_queries=self.max_queries_per_image,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            min_class_pixels=self.min_class_pixels,
            randomize=False,
            rng=random.Random(self.seed + index),  # deterministic per-sample
        )

        return (
            image_tensor,
            torch.from_numpy(train_ids.astype(np.int64)),
            torch.from_numpy(points),
            torch.from_numpy(point_labels.astype(np.int64)),
            torch.from_numpy(class_ids.astype(np.int64)),
            rel_key,
        )


def build_model(model_variant: str):
    if model_variant.lower() == "vits":
        return build_efficient_sam_vits()
    return build_efficient_sam_vitt()


def safe_torch_load(path: Path, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_checkpoint_path(checkpoint_path: Optional[str]) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    epoch_ckpt = Path("checkpoints_original/checkpoint_epoch_25.pth")
    if epoch_ckpt.exists():
        return epoch_ckpt
    return Path("checkpoints_original/best_model.pth")


def select_best_mask_per_query(pred_masks: torch.Tensor, pred_iou: torch.Tensor) -> torch.Tensor:
    _, _, _, h, w = pred_masks.shape
    best_idx = torch.argmax(pred_iou, dim=-1, keepdim=True)
    gather_index = best_idx[..., None, None].expand(-1, -1, 1, h, w)
    return torch.gather(pred_masks, dim=2, index=gather_index).squeeze(2)


def build_semantic_prediction_from_queries(
    query_mask_logits: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    probs = torch.sigmoid(query_mask_logits)
    b, q, h, w = probs.shape
    scores = torch.full((b, num_classes, h, w), -1.0, device=probs.device, dtype=probs.dtype)
    for batch_idx in range(b):
        for query_idx in range(q):
            cls = int(class_ids[batch_idx, query_idx].item())
            if 0 <= cls < num_classes:
                scores[batch_idx, cls] = torch.maximum(
                    scores[batch_idx, cls], probs[batch_idx, query_idx]
                )
    return torch.argmax(scores, dim=1)


def save_trainid_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def save_color_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation inference for paper-exact EfficientSAM semantic baseline."
    )
    parser.add_argument("--dataset-root", type=str, default="/home/aaditya/sandy/dataset")
    parser.add_argument("--val-split", type=str, default="val")
    parser.add_argument("--image-dir-name", type=str, default="img")
    parser.add_argument("--label-dir-name", type=str, default="label")
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-variant", type=str, default="vitt")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--max-queries-per-image", type=int, default=19)
    parser.add_argument("--min-class-pixels", type=int, default=16)
    parser.add_argument("--label-assume-bgr", action="store_true", default=True)
    parser.add_argument("--label-assume-rgb", action="store_false", dest="label_assume_bgr")
    parser.add_argument("--label-max-color-distance", type=float, default=55.0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="inference_outputs/val_semantic_original")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_false", dest="amp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / args.val_split / args.image_dir_name
    label_dir = dataset_root / args.val_split / args.label_dir_name
    dataset = CityscapesValPromptDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        input_size=args.input_size,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
        max_queries_per_image=args.max_queries_per_image,
        min_class_pixels=args.min_class_pixels,
        label_assume_bgr=args.label_assume_bgr,
        label_max_color_distance=args.label_max_color_distance,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.model_variant).to(device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
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
    total_pixacc = 0.0
    total_boundary_f1 = 0.0
    num_samples = 0

    progress = tqdm(dataloader, desc="Inference")
    for images, gt_labels, points, point_labels, class_ids, rel_keys in progress:
        images = images.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            pred_masks, pred_iou = model(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
            )
            query_mask_logits = select_best_mask_per_query(pred_masks, pred_iou)

        pred_labels = build_semantic_prediction_from_queries(
            query_mask_logits=query_mask_logits,
            class_ids=class_ids,
            num_classes=args.num_classes,
        )
        iou, dice = compute_semantic_iou_and_dice(
            pred_labels,
            gt_labels,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        pixacc = compute_pixel_accuracy(
            pred_labels,
            gt_labels,
            ignore_index=args.ignore_index,
        )
        gt_boundary = compute_sobel_edges_from_labels(
            gt_labels,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        pred_boundary = compute_sobel_edges_from_labels(
            pred_labels,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
        )
        boundary_f1 = boundary_f1_score(pred_boundary, gt_boundary)

        batch_size = images.shape[0]
        total_iou += float(iou.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_pixacc += float(pixacc.item()) * batch_size
        total_boundary_f1 += float(boundary_f1.item()) * batch_size
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
            miou=f"{(total_iou / max(num_samples, 1)):.4f}",
            mdice=f"{(total_dice / max(num_samples, 1)):.4f}",
            pixacc=f"{(total_pixacc / max(num_samples, 1)):.4f}",
            boundary_f1=f"{(total_boundary_f1 / max(num_samples, 1)):.4f}",
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
