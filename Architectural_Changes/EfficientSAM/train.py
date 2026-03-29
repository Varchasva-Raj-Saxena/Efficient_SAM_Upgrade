from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TrainConfig, parse_train_config
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt
from efficient_sam.losses.boundary_loss import SemanticBoundaryAwareLoss
from efficient_sam.utils.boundary_utils import boundary_f1_score, compute_sobel_edges_from_labels
from efficient_sam.utils.semantic_label_utils import decode_cityscapes_like_label_to_train_ids

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to float tensor in [0, 1] with shape (C, H, W)."""
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


class CityscapesBoundaryDataset(Dataset):
    """Cityscapes-style dataset decoded to train-id semantic masks."""

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

    def _pair_samples(self) -> List[Tuple[Path, Path]]:
        image_files = self._collect_files(self.image_dir)
        label_files = self._collect_files(self.label_dir)
        label_map = {_make_relative_key(path, self.label_dir): path for path in label_files}
        matched: List[Tuple[Path, Path]] = []
        for image_path in image_files:
            key = _make_relative_key(image_path, self.image_dir)
            label_path = label_map.get(key)
            if label_path is not None:
                matched.append((image_path, label_path))
        return matched

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _full_image_bbox_prompt(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        points = torch.tensor(
            [[[0.0, 0.0], [float(width - 1), float(height - 1)]]],
            dtype=torch.float32,
        )
        point_labels = torch.tensor([[2, 3]], dtype=torch.int64)
        return points, point_labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, label_path = self.samples[index]
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
        return image_tensor, label_tensor, points, point_labels


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("boundary_efficientsam_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


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
        return torch.tensor(0.0, device=pred_labels.device), torch.tensor(0.0, device=pred_labels.device)
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


def create_dataloaders(config: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CityscapesBoundaryDataset(
        image_dir=config.train_image_dir,
        label_dir=config.train_label_dir,
        input_size=config.input_size,
        label_assume_bgr=config.label_assume_bgr,
        label_max_color_distance=config.label_max_color_distance,
    )
    val_dataset = CityscapesBoundaryDataset(
        image_dir=config.val_image_dir,
        label_dir=config.val_label_dir,
        input_size=config.input_size,
        label_assume_bgr=config.label_assume_bgr,
        label_max_color_distance=config.label_max_color_distance,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    return train_loader, val_loader


def build_model(config: TrainConfig) -> nn.Module:
    model_variant = config.model_variant.lower()
    if model_variant == "vits":
        model = build_efficient_sam_vits(
            enable_boundary_decoder=True,
            semantic_num_classes=config.num_classes,
            enable_semantic_head=True,
        )
    else:
        model = build_efficient_sam_vitt(
            enable_boundary_decoder=True,
            semantic_num_classes=config.num_classes,
            enable_semantic_head=True,
        )
    return model


def _reduce_output(
    pred_masks: torch.Tensor,
    pred_boundaries: torch.Tensor,
    pred_semantic_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduce SAM outputs:
      masks: [B, Q, K, H, W] -> [B, 1, H, W] (auxiliary only)
      boundaries: [B, Q, K, H, W] -> [B, 1, H, W]
      semantic_logits: [B, Q, C, H, W] -> [B, C, H, W]
    """
    pred_mask_logits = pred_masks[:, 0, 0, :, :].unsqueeze(1)
    pred_boundary = pred_boundaries[:, 0, 0, :, :].unsqueeze(1)
    semantic_logits = pred_semantic_logits[:, 0, :, :, :]
    return pred_mask_logits, pred_boundary, semantic_logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: SemanticBoundaryAwareLoss,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    running_mask_loss = 0.0
    running_boundary_loss = 0.0

    progress = tqdm(loader, desc=f"Epoch [{epoch}/{config.epochs}]")
    for images, gt_labels, points, point_labels in progress:
        images = images.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)
        gt_boundary = compute_sobel_edges_from_labels(
            gt_labels,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            threshold=config.sobel_threshold,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            pred_masks, _, pred_boundaries, pred_semantic_logits = model.forward_with_boundary_and_semantics(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
                multimask_output=False,
            )
            _, pred_boundary, semantic_logits = _reduce_output(
                pred_masks, pred_boundaries, pred_semantic_logits
            )
            loss, components = criterion(
                semantic_logits=semantic_logits,
                gt_labels=gt_labels,
                pred_boundary=pred_boundary,
                gt_boundary=gt_boundary,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item())
        running_mask_loss += float(components["mask_loss"].item())
        running_boundary_loss += float(components["boundary_loss"].item())

        gpu_gb = (
            torch.cuda.memory_allocated(device=device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            mask=f"{components['mask_loss'].item():.4f}",
            boundary=f"{components['boundary_loss'].item():.4f}",
            gpu=f"{gpu_gb:.2f}GB",
        )

    num_batches = max(len(loader), 1)
    return {
        "loss": running_loss / num_batches,
        "mask_loss": running_mask_loss / num_batches,
        "boundary_loss": running_boundary_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: SemanticBoundaryAwareLoss,
    device: torch.device,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_boundary_f1 = 0.0
    total_pixacc = 0.0

    for images, gt_labels, points, point_labels in tqdm(loader, desc="Validation"):
        images = images.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)
        gt_boundary = compute_sobel_edges_from_labels(
            gt_labels,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
            threshold=config.sobel_threshold,
        )

        with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            pred_masks, _, pred_boundaries, pred_semantic_logits = model.forward_with_boundary_and_semantics(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
                multimask_output=False,
            )
            _, pred_boundary, semantic_logits = _reduce_output(
                pred_masks, pred_boundaries, pred_semantic_logits
            )
            loss, _ = criterion(
                semantic_logits=semantic_logits,
                gt_labels=gt_labels,
                pred_boundary=pred_boundary,
                gt_boundary=gt_boundary,
            )

        pred_labels = torch.argmax(semantic_logits, dim=1)
        iou, dice = compute_semantic_iou_and_dice(
            pred_labels, gt_labels, num_classes=config.num_classes, ignore_index=config.ignore_index
        )
        pixacc = compute_pixel_accuracy(pred_labels, gt_labels, ignore_index=config.ignore_index)
        pred_boundary_bin = (pred_boundary > 0.5).float()
        boundary_f1 = boundary_f1_score(pred_boundary_bin, gt_boundary)

        total_loss += float(loss.item())
        total_iou += float(iou.item())
        total_dice += float(dice.item())
        total_boundary_f1 += float(boundary_f1.item())
        total_pixacc += float(pixacc.item())

    num_batches = max(len(loader), 1)
    return {
        "loss": total_loss / num_batches,
        "iou": total_iou / num_batches,
        "dice": total_dice / num_batches,
        "boundary_f1": total_boundary_f1 / num_batches,
        "pix_acc": total_pixacc / num_batches,
    }


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loss: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def main() -> None:
    config = parse_train_config()
    seed_everything(config.seed)
    torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(checkpoint_dir / config.log_file)

    if not config.train_image_dir.exists() or not config.train_label_dir.exists():
        raise FileNotFoundError(
            f"Train dirs not found: {config.train_image_dir} and {config.train_label_dir}"
        )
    if not config.val_image_dir.exists() or not config.val_label_dir.exists():
        raise FileNotFoundError(
            f"Val dirs not found: {config.val_image_dir} and {config.val_label_dir}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_dataloaders(config, device)

    model = build_model(config).to(device)
    criterion = SemanticBoundaryAwareLoss(
        num_classes=config.num_classes,
        boundary_weight=config.boundary_loss_weight,
        ignore_index=config.ignore_index,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    best_iou = -1.0
    logger.info("Starting semantic boundary-aware training for %d epochs", config.epochs)
    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            config=config,
        )

        gpu_gb = (
            torch.cuda.memory_allocated(device=device) / (1024**3)
            if device.type == "cuda"
            else 0.0
        )
        logger.info(
            "Epoch [%d/%d] | Loss: %.4f | Mask: %.4f | Boundary: %.4f | "
            "Val mIoU: %.4f | Val mDice: %.4f | Val PixAcc: %.4f | "
            "Val Boundary-F1: %.4f | GPU: %.2fGB",
            epoch,
            config.epochs,
            train_stats["loss"],
            train_stats["mask_loss"],
            train_stats["boundary_loss"],
            val_stats["iou"],
            val_stats["dice"],
            val_stats["pix_acc"],
            val_stats["boundary_f1"],
            gpu_gb,
        )

        if epoch % config.checkpoint_every == 0:
            save_checkpoint(
                path=checkpoint_dir / f"checkpoint_epoch_{epoch}.pth",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss=val_stats["loss"],
            )

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            save_checkpoint(
                path=checkpoint_dir / config.best_model_name,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loss=val_stats["loss"],
            )
            logger.info("Saved new best checkpoint at epoch %d with mIoU %.4f", epoch, best_iou)

    logger.info("Training completed. Best mIoU: %.4f", best_iou)


if __name__ == "__main__":
    main()
