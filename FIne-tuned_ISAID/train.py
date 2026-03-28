

import argparse
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloaders
from model_setup import (
    load_efficient_sam,
    freeze_model,
    inject_lora,
    get_trainable_params,
)


# ===================================================================
# Loss Functions
# ===================================================================

def sigmoid_focal_loss(inputs: torch.Tensor,
                       targets: torch.Tensor,
                       alpha: float = 0.25,
                       gamma: float = 2.0) -> torch.Tensor:
    """
    Sigmoid Focal Loss — down-weights well-classified examples.

    Args:
        inputs:  raw logits [B, 1, H, W]
        targets: binary ground truth [B, 1, H, W]
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets,
                                                  reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * focal_weight * ce_loss
    return loss.mean()


def dice_loss(inputs: torch.Tensor,
              targets: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """
    Dice Loss for binary segmentation.

    Args:
        inputs:  raw logits [B, 1, H, W]
        targets: binary ground truth [B, 1, H, W]
    """
    probs = torch.sigmoid(inputs)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


def combined_loss(pred_logits: torch.Tensor,
                  gt_masks: torch.Tensor) -> torch.Tensor:
    """
    SAM-style combined loss:  20 × Focal  +  1 × Dice
    """
    focal = sigmoid_focal_loss(pred_logits, gt_masks)
    dice = dice_loss(pred_logits, gt_masks)
    return 20.0 * focal + 1.0 * dice


# ===================================================================
# Metrics
# ===================================================================

def compute_iou(pred_logits: torch.Tensor,
                gt_masks: torch.Tensor,
                threshold: float = 0.0) -> float:
    """
    Compute mean Intersection-over-Union for a batch.

    Args:
        pred_logits: raw logits [B, 1, H, W]
        gt_masks:    binary masks [B, 1, H, W]
        threshold:   logit threshold (EfficientSAM uses 0.0)

    Returns:
        Mean IoU (float).
    """
    preds = (pred_logits > threshold).float()
    intersection = (preds * gt_masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


# ===================================================================
# Training & Validation Steps
# ===================================================================

def train_one_epoch(model, dataloader, optimizer, device, scaler,
                    max_batches: int = 0):
    """Run one training epoch with AMP.  Returns average loss."""
    model.train()
    # Keep encoder in eval mode (BatchNorm / dropout if any)
    model.image_encoder.eval()
    model.prompt_encoder.eval()

    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, gt_masks, bbox_points, bbox_labels) in enumerate(
            tqdm(dataloader, desc="  Train", leave=False), start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break

        images = images.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)
        bbox_points = bbox_points.to(device, non_blocking=True)
        bbox_labels = bbox_labels.to(device, non_blocking=True)

        # Forward — frozen encoder (AMP for speed)
        with torch.no_grad(), autocast(device_type="cuda"):
            image_embeddings = model.get_image_embeddings(images)

        # Predict masks via decoder (AMP)
        with autocast(device_type="cuda"):
            pred_masks, iou_preds = model.predict_masks(
                image_embeddings=image_embeddings,
                batched_points=bbox_points,         # [B, 1, 2, 2]
                batched_point_labels=bbox_labels,    # [B, 1, 2]
                multimask_output=False,
                input_h=1024,
                input_w=1024,
            )
            # pred_masks shape: [B, 1, 1, 256, 256]
            pred_masks = pred_masks.squeeze(1)  # [B, 1, 256, 256]
            loss = combined_loss(pred_masks, gt_masks)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device, max_batches: int = 0):
    """Run validation with AMP.  Returns (avg_loss, mean_iou)."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    num_batches = 0

    for batch_idx, (images, gt_masks, bbox_points, bbox_labels) in enumerate(
            tqdm(dataloader, desc="  Val  ", leave=False), start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break

        images = images.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)
        bbox_points = bbox_points.to(device, non_blocking=True)
        bbox_labels = bbox_labels.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            image_embeddings = model.get_image_embeddings(images)
            pred_masks, iou_preds = model.predict_masks(
                image_embeddings=image_embeddings,
                batched_points=bbox_points,
                batched_point_labels=bbox_labels,
                multimask_output=False,
                input_h=1024,
                input_w=1024,
            )
            pred_masks = pred_masks.squeeze(1)
            loss = combined_loss(pred_masks, gt_masks)

        iou = compute_iou(pred_masks, gt_masks)

        running_loss += loss.item()
        running_iou += iou
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    avg_iou = running_iou / max(num_batches, 1)
    return avg_loss, avg_iou


# ===================================================================
# Main
# ===================================================================

def ensure_patched_dataset(data_root: str) -> None:
    """
    Create patched dataset automatically if annotations are missing.
    """
    train_ann = os.path.join(data_root, "train", "annotations.json")
    val_ann = os.path.join(data_root, "val", "annotations.json")
    if os.path.isfile(train_ann) and os.path.isfile(val_ann):
        return

    if data_root.endswith("_patched"):
        raw_root = data_root[:-8]
    else:
        raw_root = "dataset"

    if not os.path.isdir(raw_root):
        raise FileNotFoundError(
            f"Missing patched annotations under '{data_root}', and raw dataset "
            f"root '{raw_root}' was not found. Please run prepare_data.py first."
        )

    print(
        f"⚙️  Missing patched annotations. Generating '{data_root}' from "
        f"'{raw_root}' ..."
    )
    cmd = [
        sys.executable,
        "prepare_data.py",
        "--data_root",
        raw_root,
        "--output_root",
        data_root,
    ]
    subprocess.run(cmd, check=True)

    if not (os.path.isfile(train_ann) and os.path.isfile(val_ann)):
        raise RuntimeError(
            "Data preparation finished but annotations.json files were not "
            "created as expected."
        )

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientSAM on patched iSAID data")
    parser.add_argument("--data_root", type=str, default="dataset_patched")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="vits",
                        choices=["vits", "vitt"])
    parser.add_argument("--use_lora", action="store_true",
                        help="Inject LoRA into the frozen image encoder")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (auto-detected if omitted)")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume from")
    parser.add_argument("--extra_epochs", type=int, default=0,
                        help="If resuming, train this many more epochs beyond the checkpoint epoch")
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="If >0, cap train batches per epoch for faster runs")
    parser.add_argument("--max_val_batches", type=int, default=0,
                        help="If >0, cap val batches per epoch for faster runs")
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"🖥️  Device: {device}")

    # Performance optimizations
    torch.backends.cudnn.benchmark = True

    # Ensure patched dataset exists
    ensure_patched_dataset(args.data_root)

    # Model
    model = load_efficient_sam(args.model_type, device=device)
    freeze_model(model)
    if args.use_lora:
        inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model = model.to(device)

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers)

    # Optimizer & scheduler
    trainable_params = get_trainable_params(model)
    optimizer = AdamW(trainable_params, lr=args.lr,
                      weight_decay=args.weight_decay)
    scaler = GradScaler()

    # Checkpointing
    os.makedirs(args.save_dir, exist_ok=True)
    best_iou = 0.0
    completed_epoch = 0
    total_epochs = args.epochs

    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.mask_decoder.load_state_dict(ckpt["mask_decoder"])

        if args.use_lora:
            lora_state = ckpt.get("lora_params")
            if not lora_state:
                raise RuntimeError(
                    "The resume checkpoint does not contain LoRA weights. "
                    "This run was started with --use_lora, so it cannot be "
                    "resumed faithfully from that checkpoint."
                )
            for name, param in model.image_encoder.named_parameters():
                if name in lora_state:
                    param.data.copy_(lora_state[name].to(device))

        completed_epoch = int(ckpt.get("epoch", 0))
        best_iou = float(ckpt.get("best_iou", ckpt.get("val_iou", 0.0)))
        if args.extra_epochs > 0:
            total_epochs = completed_epoch + args.extra_epochs

        print(f"🔁 Resuming from {args.resume_checkpoint}")
        print(f"   Loaded checkpoint epoch: {completed_epoch}")

    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    if args.resume_checkpoint:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print("   ℹ️  Optimizer state not found; continuing with a fresh optimizer.")

        if "scheduler" in ckpt:
            scheduler_state = ckpt["scheduler"]
            scheduler_state["T_max"] = total_epochs
            scheduler.load_state_dict(scheduler_state)
        elif completed_epoch > 0:
            scheduler.last_epoch = completed_epoch - 1
            scheduler.step()
            print("   ℹ️  Scheduler state not found; approximated scheduler position from the checkpoint epoch.")

    print(f"\n{'='*60}")
    print(f"  Training EfficientSAM-{args.model_type.upper()} for "
          f"{total_epochs} epochs")
    print(f"  Batch size: {args.batch_size}  |  LR: {args.lr}  "
          f"|  LoRA: {'ON' if args.use_lora else 'OFF'}")
    if args.max_train_batches > 0 or args.max_val_batches > 0:
        print(f"  Batch caps -> train: {args.max_train_batches or 'full'}  "
              f"|  val: {args.max_val_batches or 'full'}")
    print(f"{'='*60}\n")

    for epoch in range(completed_epoch + 1, total_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scaler,
            max_batches=args.max_train_batches,
        )
        val_loss, val_iou = validate(
            model, val_loader, device,
            max_batches=args.max_val_batches,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:3d}/{total_epochs}  |  "
              f"Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val IoU: {val_iou:.4f}  |  "
              f"LR: {lr_now:.2e}  |  "
              f"Time: {elapsed:.0f}s")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt_path = os.path.join(args.save_dir, "best_mask_decoder.pth")
            # Save mask decoder + LoRA params if applicable
            state = {
                "epoch": epoch,
                "val_iou": val_iou,
                "best_iou": best_iou,
                "mask_decoder": model.mask_decoder.state_dict(),
                "model_type": args.model_type,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
            }
            if args.use_lora:
                # Also save LoRA parameters from encoder
                lora_state = {}
                for name, param in model.image_encoder.named_parameters():
                    if param.requires_grad:
                        lora_state[name] = param.data.clone()
                state["lora_params"] = lora_state
            torch.save(state, ckpt_path)
            print(f"  ✅ Saved best model (IoU={val_iou:.4f}) → {ckpt_path}")

        # Save latest checkpoint
        torch.save({
            "epoch": epoch,
            "val_iou": val_iou,
            "best_iou": best_iou,
            "mask_decoder": model.mask_decoder.state_dict(),
            "model_type": args.model_type,
            "use_lora": args.use_lora,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "lora_params": {
                name: param.data.clone()
                for name, param in model.image_encoder.named_parameters()
                if param.requires_grad
            } if args.use_lora else {},
        }, os.path.join(args.save_dir, "latest_checkpoint.pth"))

    print(f"\n🏁 Training complete! Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
