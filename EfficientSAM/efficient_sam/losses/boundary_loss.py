from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def dice_loss_from_logits(
    pred_mask_logits: torch.Tensor, gt_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Dice loss on mask logits."""
    pred_prob = torch.sigmoid(pred_mask_logits)
    intersection = (pred_prob * gt_mask).sum(dim=(1, 2, 3))
    denominator = pred_prob.sum(dim=(1, 2, 3)) + gt_mask.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + eps) / (denominator + eps))
    return dice.mean()


class BoundaryAwareLoss(nn.Module):
    """Combined segmentation and boundary-aware objective."""

    def __init__(self, boundary_weight: float = 0.5) -> None:
        super().__init__()
        self.boundary_weight = boundary_weight
        self.mask_bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_mask_logits: torch.Tensor,
        gt_mask: torch.Tensor,
        pred_boundary: torch.Tensor,
        gt_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_bce_loss = self.mask_bce(pred_mask_logits, gt_mask)
        mask_dice_loss = dice_loss_from_logits(pred_mask_logits, gt_mask)
        mask_loss = mask_bce_loss + mask_dice_loss

        boundary_pred = pred_boundary.clamp(min=1e-6, max=1.0 - 1e-6)
        device_type = boundary_pred.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            boundary_loss = F.binary_cross_entropy(
                boundary_pred.float(), gt_boundary.float()
            )

        total_loss = mask_loss + self.boundary_weight * boundary_loss
        components = {
            "mask_loss": mask_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
        }
        return total_loss, components


def multiclass_dice_loss_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Multi-class Dice loss with ignore index support.

    Args:
        logits: (B, C, H, W)
        target: (B, H, W) integer labels
    """
    probs = torch.softmax(logits, dim=1)
    valid = target != ignore_index
    clamped = torch.clamp(target, 0, num_classes - 1)
    one_hot = F.one_hot(clamped.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice_per_class = (2.0 * intersection + eps) / (denominator + eps)

    present = one_hot.sum(dim=(0, 2, 3)) > 0
    if present.any():
        return 1.0 - dice_per_class[present].mean()
    return 1.0 - dice_per_class.mean()


class SemanticBoundaryAwareLoss(nn.Module):
    """Semantic segmentation loss + boundary supervision."""

    def __init__(
        self,
        num_classes: int,
        boundary_weight: float = 0.5,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.seg_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        semantic_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        pred_boundary: torch.Tensor,
        gt_boundary: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seg_ce_loss = self.seg_ce(semantic_logits, gt_labels.long())
        seg_dice_loss = multiclass_dice_loss_from_logits(
            semantic_logits,
            gt_labels,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )
        seg_loss = seg_ce_loss + seg_dice_loss

        boundary_pred = pred_boundary.clamp(min=1e-6, max=1.0 - 1e-6)
        device_type = boundary_pred.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            boundary_loss = F.binary_cross_entropy(
                boundary_pred.float(), gt_boundary.float()
            )

        total_loss = seg_loss + self.boundary_weight * boundary_loss
        components = {
            "mask_loss": seg_loss.detach(),
            "boundary_loss": boundary_loss.detach(),
        }
        return total_loss, components
