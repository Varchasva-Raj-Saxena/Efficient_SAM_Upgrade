from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cityscapes_semantic_utils import (
    boundary_f1_score,
    build_prompt_targets_from_train_ids,
    compute_pixel_accuracy,
    compute_semantic_iou_and_dice,
    compute_sobel_edges_from_labels,
    decode_cityscapes_like_label_to_train_ids,
    limit_samples,
    pair_image_label_files,
)
from config import TrainConfig, parse_train_config
from efficient_sam.build_efficient_sam import build_efficient_sam_vits, build_efficient_sam_vitt


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CityscapesPromptDataset(Dataset):
    """Class-prompted dataset for paper-exact EfficientSAM semantic training."""

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
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label_path, _ = self.samples[index]
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

        (
            points,
            point_labels,
            target_masks,
            valid_masks,
            class_ids,
        ) = build_prompt_targets_from_train_ids(
            train_ids=train_ids,
            max_queries=self.max_queries_per_image,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            min_class_pixels=self.min_class_pixels,
            randomize=False,
            rng=self.rng,
        )

        return (
            image_tensor,
            torch.from_numpy(train_ids.astype(np.int64)),
            torch.from_numpy(points),
            torch.from_numpy(point_labels.astype(np.int64)),
            torch.from_numpy(target_masks),
            torch.from_numpy(valid_masks),
            torch.from_numpy(class_ids.astype(np.int64)),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("original_efficientsam_cityscapes_semantic")
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


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = valid_mask.sum().clamp_min(1.0)
    return (loss * valid_mask).sum() / denom


def masked_dice_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits) * valid_mask
    targets = targets * valid_mask
    intersection = (probs * targets).sum(dim=(2, 3))
    denominator = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    query_has_valid = valid_mask.sum(dim=(2, 3)) > 0
    if query_has_valid.any():
        return dice[query_has_valid].mean()
    return logits.new_tensor(0.0)


def select_best_mask_per_query(pred_masks: torch.Tensor, pred_iou: torch.Tensor) -> torch.Tensor:
    """Select the best logit mask per query using decoder IoU scores."""
    _, _, _, h, w = pred_masks.shape
    best_idx = torch.argmax(pred_iou, dim=-1, keepdim=True)  # [B, Q, 1]
    gather_index = best_idx[..., None, None].expand(-1, -1, 1, h, w)
    return torch.gather(pred_masks, dim=2, index=gather_index).squeeze(2)  # [B, Q, H, W]


def build_semantic_prediction_from_queries(
    query_mask_logits: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    """Merge class-prompted query masks into a semantic trainId map."""
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


def create_dataloaders(config: TrainConfig, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    train_dataset = CityscapesPromptDataset(
        image_dir=config.train_image_dir,
        label_dir=config.train_label_dir,
        input_size=config.input_size,
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        max_queries_per_image=config.max_queries_per_image,
        min_class_pixels=config.min_class_pixels,
        label_assume_bgr=config.label_assume_bgr,
        label_max_color_distance=config.label_max_color_distance,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    val_dataset = CityscapesPromptDataset(
        image_dir=config.val_image_dir,
        label_dir=config.val_label_dir,
        input_size=config.input_size,
        num_classes=config.num_classes,
        ignore_index=config.ignore_index,
        max_queries_per_image=config.max_queries_per_image,
        min_class_pixels=config.min_class_pixels,
        label_assume_bgr=config.label_assume_bgr,
        label_max_color_distance=config.label_max_color_distance,
        max_samples=config.max_val_samples,
        seed=config.seed,
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
    if config.model_variant.lower() == "vits":
        return build_efficient_sam_vits()
    return build_efficient_sam_vitt()


def compute_loss(
    query_mask_logits: torch.Tensor,
    target_masks: torch.Tensor,
    valid_masks: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    bce = masked_bce_with_logits(query_mask_logits, target_masks, valid_masks)
    dice = masked_dice_from_logits(query_mask_logits, target_masks, valid_masks)
    total = bce + dice
    return total, {"bce": bce.detach(), "dice": dice.detach()}


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainConfig,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0

    progress = tqdm(loader, desc=f"Epoch [{epoch}/{config.epochs}]")
    for (
        images,
        _gt_labels,
        points,
        point_labels,
        target_masks,
        valid_masks,
        _class_ids,
    ) in progress:
        images = images.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)
        target_masks = target_masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            pred_masks, pred_iou = model(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
            )
            query_mask_logits = select_best_mask_per_query(pred_masks, pred_iou)
            loss, components = compute_loss(
                query_mask_logits=query_mask_logits,
                target_masks=target_masks,
                valid_masks=valid_masks,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_bce += float(components["bce"].item())
        total_dice += float(components["dice"].item())
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            bce=f"{components['bce'].item():.4f}",
            dice=f"{components['dice'].item():.4f}",
        )

    num_batches = max(len(loader), 1)
    return {
        "loss": total_loss / num_batches,
        "bce": total_bce / num_batches,
        "dice": total_dice / num_batches,
    }


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_boundary_f1 = 0.0
    total_pixacc = 0.0
    num_samples = 0

    for (
        images,
        gt_labels,
        points,
        point_labels,
        target_masks,
        valid_masks,
        class_ids,
    ) in tqdm(loader, desc="Validation"):
        images = images.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        points = points.to(device, non_blocking=True)
        point_labels = point_labels.to(device, non_blocking=True)
        target_masks = target_masks.to(device, non_blocking=True)
        valid_masks = valid_masks.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
            pred_masks, pred_iou = model(
                images,
                points,
                point_labels,
                scale_to_original_image_size=True,
            )
            query_mask_logits = select_best_mask_per_query(pred_masks, pred_iou)
            loss, _ = compute_loss(
                query_mask_logits=query_mask_logits,
                target_masks=target_masks,
                valid_masks=valid_masks,
            )

        pred_labels = build_semantic_prediction_from_queries(
            query_mask_logits=query_mask_logits,
            class_ids=class_ids,
            num_classes=config.num_classes,
        )
        iou, dice = compute_semantic_iou_and_dice(
            pred_labels,
            gt_labels,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        pixacc = compute_pixel_accuracy(
            pred_labels,
            gt_labels,
            ignore_index=config.ignore_index,
        )
        gt_boundary = compute_sobel_edges_from_labels(
            gt_labels,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        pred_boundary = compute_sobel_edges_from_labels(
            pred_labels,
            num_classes=config.num_classes,
            ignore_index=config.ignore_index,
        )
        boundary_f1 = boundary_f1_score(pred_boundary, gt_boundary)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_iou += float(iou.item()) * batch_size
        total_dice += float(dice.item()) * batch_size
        total_pixacc += float(pixacc.item()) * batch_size
        total_boundary_f1 += float(boundary_f1.item()) * batch_size
        num_samples += batch_size

    denom = max(num_samples, 1)
    return {
        "loss": total_loss / denom,
        "iou": total_iou / denom,
        "dice": total_dice / denom,
        "pix_acc": total_pixacc / denom,
        "boundary_f1": total_boundary_f1 / denom,
    }


def save_checkpoint(
    *,
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
    train_loader, val_loader = create_dataloaders(config=config, device=device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    best_iou = -1.0
    logger.info(
        "Starting ORIGINAL EfficientSAM (paper-arch) class-prompted semantic training for %d epochs",
        config.epochs,
    )
    for epoch in range(1, config.epochs + 1):
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            config=config,
            epoch=epoch,
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            device=device,
            config=config,
        )
        logger.info(
            "Epoch [%d/%d] | Loss: %.4f (BCE: %.4f, Dice: %.4f) | "
            "Val mIoU: %.4f | Val mDice: %.4f | Val PixAcc: %.4f | Val Boundary-F1: %.4f",
            epoch,
            config.epochs,
            train_stats["loss"],
            train_stats["bce"],
            train_stats["dice"],
            val_stats["iou"],
            val_stats["dice"],
            val_stats["pix_acc"],
            val_stats["boundary_f1"],
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
