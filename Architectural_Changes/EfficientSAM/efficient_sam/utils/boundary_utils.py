from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return sobel_x, sobel_y


def compute_soft_sobel_magnitude(mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute normalized Sobel gradient magnitude for binary masks.

    Args:
        mask: Tensor with shape (B, 1, H, W).
        eps: Small epsilon for stability.

    Returns:
        Tensor with shape (B, 1, H, W) in [0, 1].
    """
    if mask.dim() != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask with shape (B, 1, H, W), got {tuple(mask.shape)}")

    sobel_x, sobel_y = _sobel_kernels(mask.device, mask.dtype)
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)

    flat = magnitude.flatten(start_dim=1)
    max_vals = flat.max(dim=1).values.view(-1, 1, 1, 1).clamp_min(eps)
    return magnitude / max_vals


@torch.no_grad()
def compute_sobel_edges(mask: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Create binary boundary maps using Sobel gradients.

    Args:
        mask: Tensor with shape (B, 1, H, W).
        threshold: Threshold on normalized gradient magnitude.

    Returns:
        Binary boundary tensor with shape (B, 1, H, W).
    """
    normalized_magnitude = compute_soft_sobel_magnitude(mask.float())
    return (normalized_magnitude > threshold).to(mask.dtype)


@torch.no_grad()
def compute_sobel_edges_from_labels(
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    threshold: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute binary boundaries from multi-class label maps using Sobel on one-hot masks.

    Args:
        labels: Tensor (B, H, W), integer train IDs.
        num_classes: Number of semantic classes.
        ignore_index: Label value to ignore.
        threshold: Threshold on normalized max class-gradient magnitude.
        eps: Numerical stability epsilon.

    Returns:
        Binary boundary map (B, 1, H, W).
    """
    if labels.dim() != 3:
        raise ValueError(f"Expected labels with shape (B, H, W), got {tuple(labels.shape)}")

    valid = labels != ignore_index
    clamped = torch.clamp(labels, 0, num_classes - 1)
    one_hot = F.one_hot(clamped.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    one_hot = one_hot * valid.unsqueeze(1).float()

    sobel_x, sobel_y = _sobel_kernels(one_hot.device, one_hot.dtype)
    sobel_x = sobel_x.repeat(num_classes, 1, 1, 1)
    sobel_y = sobel_y.repeat(num_classes, 1, 1, 1)

    grad_x = F.conv2d(one_hot, sobel_x, padding=1, groups=num_classes)
    grad_y = F.conv2d(one_hot, sobel_y, padding=1, groups=num_classes)
    magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + eps)
    max_magnitude = magnitude.max(dim=1, keepdim=True).values

    flat = max_magnitude.flatten(start_dim=1)
    max_vals = flat.max(dim=1).values.view(-1, 1, 1, 1).clamp_min(eps)
    normalized = max_magnitude / max_vals
    return (normalized > threshold).float()


@torch.no_grad()
def boundary_f1_score(
    pred_boundary: torch.Tensor, gt_boundary: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute mean boundary F1 score across a batch.

    Args:
        pred_boundary: Predicted boundary map (B, 1, H, W), in {0,1} or [0,1].
        gt_boundary: Ground-truth boundary map (B, 1, H, W), in {0,1}.
        eps: Numerical stability value.
    """
    pred = (pred_boundary > 0.5).float()
    gt = (gt_boundary > 0.5).float()

    true_positive = (pred * gt).sum(dim=(1, 2, 3))
    precision = (true_positive + eps) / (pred.sum(dim=(1, 2, 3)) + eps)
    recall = (true_positive + eps) / (gt.sum(dim=(1, 2, 3)) + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    return f1.mean()
