from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

CITYSCAPES_LABEL_COLORS_RGB = np.array(
    [
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (111, 74, 0),
        (81, 0, 81),
        (128, 64, 128),
        (244, 35, 232),
        (250, 170, 160),
        (230, 150, 140),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (180, 165, 180),
        (150, 100, 100),
        (150, 120, 90),
        (153, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 0, 90),
        (0, 0, 110),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.float32,
)

CITYSCAPES_LABEL_TO_TRAINID = np.array(
    [
        255,
        255,
        255,
        255,
        255,
        255,
        255,
        0,
        1,
        255,
        255,
        2,
        3,
        4,
        255,
        255,
        255,
        5,
        255,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        255,
        255,
        16,
        17,
        18,
    ],
    dtype=np.uint8,
)

CITYSCAPES_TRAINID_COLORS_RGB = np.array(
    [
        (128, 64, 128),
        (244, 35, 232),
        (70, 70, 70),
        (102, 102, 156),
        (190, 153, 153),
        (153, 153, 153),
        (250, 170, 30),
        (220, 220, 0),
        (107, 142, 35),
        (152, 251, 152),
        (70, 130, 180),
        (220, 20, 60),
        (255, 0, 0),
        (0, 0, 142),
        (0, 0, 70),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 230),
        (119, 11, 32),
    ],
    dtype=np.uint8,
)


def _nearest_cityscapes_label_ids(colors_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    diff = colors_rgb[:, None, :] - CITYSCAPES_LABEL_COLORS_RGB[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest_label_ids = np.argmin(dist2, axis=1)
    nearest_dist = np.sqrt(np.min(dist2, axis=1))
    return nearest_label_ids, nearest_dist


def _orientation_error(colors_rgb: np.ndarray, assume_bgr: bool) -> float:
    oriented = colors_rgb[:, ::-1] if assume_bgr else colors_rgb
    _, dist = _nearest_cityscapes_label_ids(oriented.astype(np.float32))
    return float(np.mean(dist))


def decode_cityscapes_like_label_to_train_ids(
    label_rgb: np.ndarray,
    *,
    assume_bgr: Optional[bool] = None,
    max_color_distance: float = 55.0,
) -> np.ndarray:
    if label_rgb.ndim != 3 or label_rgb.shape[2] != 3:
        raise ValueError(f"Expected label_rgb shape (H, W, 3), got {label_rgb.shape}")

    colors = label_rgb.astype(np.float32).reshape(-1, 3)
    if assume_bgr is None:
        stride = max(1, len(colors) // 20000)
        rgb_err = _orientation_error(colors[::stride], assume_bgr=False)
        bgr_err = _orientation_error(colors[::stride], assume_bgr=True)
        assume_bgr = bgr_err < rgb_err

    oriented = colors[:, ::-1] if assume_bgr else colors
    label_ids, dist = _nearest_cityscapes_label_ids(oriented.astype(np.float32))
    train_ids = CITYSCAPES_LABEL_TO_TRAINID[label_ids].astype(np.uint8)
    if max_color_distance > 0:
        train_ids[dist > max_color_distance] = 255
    return train_ids.reshape(label_rgb.shape[0], label_rgb.shape[1])


def encode_train_ids_to_color(
    train_ids: np.ndarray,
    *,
    bgr_output: bool = True,
    ignore_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if train_ids.ndim != 2:
        raise ValueError(f"Expected train_ids shape (H, W), got {train_ids.shape}")

    h, w = train_ids.shape
    output = np.zeros((h, w, 3), dtype=np.uint8)
    output[:] = np.array(ignore_color, dtype=np.uint8)
    valid = train_ids != 255
    clipped = np.clip(train_ids, 0, CITYSCAPES_TRAINID_COLORS_RGB.shape[0] - 1)
    output[valid] = CITYSCAPES_TRAINID_COLORS_RGB[clipped[valid]]
    if bgr_output:
        output = output[:, :, ::-1]
    return output


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


def pair_image_label_files(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path, str]]:
    image_files = sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    label_files = sorted(
        [
            path
            for path in label_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    label_map = {_make_relative_key(path, label_dir): path for path in label_files}
    matched: List[Tuple[Path, Path, str]] = []
    for image_path in image_files:
        rel_key = _make_relative_key(image_path, image_dir)
        label_path = label_map.get(rel_key)
        if label_path is not None:
            matched.append((image_path, label_path, rel_key))
    return matched


def build_prompt_targets_from_train_ids(
    train_ids: np.ndarray,
    *,
    max_queries: int,
    num_classes: int,
    ignore_index: int,
    min_class_pixels: int,
    randomize: bool,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if train_ids.ndim != 2:
        raise ValueError(f"Expected train_ids shape (H, W), got {train_ids.shape}")
    if max_queries <= 0:
        raise ValueError("max_queries must be > 0")

    h, w = train_ids.shape
    valid_pixels = (train_ids != ignore_index).astype(np.float32)
    unique_classes = np.unique(train_ids)
    candidate_classes = [
        int(cls)
        for cls in unique_classes
        if 0 <= int(cls) < num_classes
        and int(cls) != ignore_index
        and int(np.count_nonzero(train_ids == int(cls))) >= min_class_pixels
    ]
    if randomize and len(candidate_classes) > max_queries:
        candidate_classes = rng.sample(candidate_classes, k=max_queries)
    else:
        candidate_classes = sorted(candidate_classes)[:max_queries]

    points = np.full((max_queries, 2, 2), -1.0, dtype=np.float32)
    point_labels = np.full((max_queries, 2), -1, dtype=np.int64)
    target_masks = np.zeros((max_queries, h, w), dtype=np.float32)
    valid_masks = np.zeros((max_queries, h, w), dtype=np.float32)
    class_ids = np.full((max_queries,), ignore_index, dtype=np.int64)

    for query_idx, cls in enumerate(candidate_classes):
        ys, xs = np.nonzero(train_ids == cls)
        if ys.size == 0:
            continue
        points[query_idx, 0] = np.array([float(xs.min()), float(ys.min())], dtype=np.float32)
        points[query_idx, 1] = np.array([float(xs.max()), float(ys.max())], dtype=np.float32)
        point_labels[query_idx] = np.array([2, 3], dtype=np.int64)
        target_masks[query_idx] = (train_ids == cls).astype(np.float32)
        valid_masks[query_idx] = valid_pixels
        class_ids[query_idx] = int(cls)
    return points, point_labels, target_masks, valid_masks, class_ids


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


@torch.no_grad()
def compute_sobel_edges_from_labels(
    labels: torch.Tensor,
    *,
    num_classes: int,
    ignore_index: int = 255,
    threshold: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
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


def compute_semantic_iou_and_dice(
    pred_labels: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
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
    *,
    ignore_index: int = 255,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = gt_labels != ignore_index
    correct = (pred_labels == gt_labels) & valid
    return (correct.sum().float() + eps) / (valid.sum().float() + eps)


@torch.no_grad()
def boundary_f1_score(
    pred_boundary: torch.Tensor,
    gt_boundary: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    pred = (pred_boundary > 0.5).float()
    gt = (gt_boundary > 0.5).float()
    true_positive = (pred * gt).sum(dim=(1, 2, 3))
    precision = (true_positive + eps) / (pred.sum(dim=(1, 2, 3)) + eps)
    recall = (true_positive + eps) / (gt.sum(dim=(1, 2, 3)) + eps)
    return ((2.0 * precision * recall + eps) / (precision + recall + eps)).mean()


def limit_samples(samples: Sequence[Tuple[Path, Path, str]], max_samples: int) -> List[Tuple[Path, Path, str]]:
    if max_samples is None or max_samples <= 0:
        return list(samples)
    return list(samples[:max_samples])
