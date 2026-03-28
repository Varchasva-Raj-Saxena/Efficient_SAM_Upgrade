from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Cityscapes label colors (RGB) and trainId mapping.
# trainId 255 denotes ignore.
CITYSCAPES_LABEL_COLORS_RGB = np.array(
    [
        (0, 0, 0),      # 0 unlabeled
        (0, 0, 0),      # 1 ego vehicle
        (0, 0, 0),      # 2 rectification border
        (0, 0, 0),      # 3 out of roi
        (0, 0, 0),      # 4 static
        (111, 74, 0),   # 5 dynamic
        (81, 0, 81),    # 6 ground
        (128, 64, 128), # 7 road
        (244, 35, 232), # 8 sidewalk
        (250, 170, 160),# 9 parking
        (230, 150, 140),# 10 rail track
        (70, 70, 70),   # 11 building
        (102, 102, 156),# 12 wall
        (190, 153, 153),# 13 fence
        (180, 165, 180),# 14 guard rail
        (150, 100, 100),# 15 bridge
        (150, 120, 90), # 16 tunnel
        (153, 153, 153),# 17 pole
        (153, 153, 153),# 18 polegroup
        (250, 170, 30), # 19 traffic light
        (220, 220, 0),  # 20 traffic sign
        (107, 142, 35), # 21 vegetation
        (152, 251, 152),# 22 terrain
        (70, 130, 180), # 23 sky
        (220, 20, 60),  # 24 person
        (255, 0, 0),    # 25 rider
        (0, 0, 142),    # 26 car
        (0, 0, 70),     # 27 truck
        (0, 60, 100),   # 28 bus
        (0, 0, 90),     # 29 caravan
        (0, 0, 110),    # 30 trailer
        (0, 80, 100),   # 31 train
        (0, 0, 230),    # 32 motorcycle
        (119, 11, 32),  # 33 bicycle
    ],
    dtype=np.float32,
)

CITYSCAPES_LABEL_TO_TRAINID = np.array(
    [
        255, 255, 255, 255, 255, 255, 255,
        0, 1, 255, 255, 2, 3, 4, 255, 255, 255,
        5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        15, 255, 255, 16, 17, 18,
    ],
    dtype=np.uint8,
)

# Standard Cityscapes trainId colors in RGB.
CITYSCAPES_TRAINID_COLORS_RGB = np.array(
    [
        (128, 64, 128),  # road
        (244, 35, 232),  # sidewalk
        (70, 70, 70),    # building
        (102, 102, 156), # wall
        (190, 153, 153), # fence
        (153, 153, 153), # pole
        (250, 170, 30),  # traffic light
        (220, 220, 0),   # traffic sign
        (107, 142, 35),  # vegetation
        (152, 251, 152), # terrain
        (70, 130, 180),  # sky
        (220, 20, 60),   # person
        (255, 0, 0),     # rider
        (0, 0, 142),     # car
        (0, 0, 70),      # truck
        (0, 60, 100),    # bus
        (0, 80, 100),    # train
        (0, 0, 230),     # motorcycle
        (119, 11, 32),   # bicycle
    ],
    dtype=np.uint8,
)


def _nearest_cityscapes_label_ids(colors_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map colors to nearest Cityscapes label-id colors.

    Args:
        colors_rgb: Array with shape (N, 3), float32.

    Returns:
        nearest_label_ids: (N,) label-id indices [0, 33]
        nearest_dist: (N,) Euclidean color distance
    """
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
    """
    Decode a noisy Cityscapes-like color label map into train IDs.

    This handles color jitter/compression by nearest-color matching and supports
    datasets where label colors were stored in BGR order.

    Args:
        label_rgb: (H, W, 3), uint8.
        assume_bgr: If None, auto-detect orientation by nearest-color error.
        max_color_distance: Pixels beyond this distance are set to ignore (255).

    Returns:
        train_ids: (H, W), uint8 in [0, 18] or 255 for ignore.
    """
    if label_rgb.ndim != 3 or label_rgb.shape[2] != 3:
        raise ValueError(f"Expected label_rgb shape (H, W, 3), got {label_rgb.shape}")

    colors = label_rgb.astype(np.float32).reshape(-1, 3)
    if assume_bgr is None:
        rgb_err = _orientation_error(colors[:: max(1, len(colors) // 20000)], assume_bgr=False)
        bgr_err = _orientation_error(colors[:: max(1, len(colors) // 20000)], assume_bgr=True)
        assume_bgr = bgr_err < rgb_err

    oriented = colors[:, ::-1] if assume_bgr else colors
    label_ids, dist = _nearest_cityscapes_label_ids(oriented.astype(np.float32))
    train_ids = CITYSCAPES_LABEL_TO_TRAINID[label_ids].astype(np.uint8)
    if max_color_distance > 0:
        train_ids[dist > max_color_distance] = 255
    return train_ids.reshape(label_rgb.shape[0], label_rgb.shape[1])


def encode_train_ids_to_color(
    train_ids: np.ndarray, *, bgr_output: bool = True, ignore_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    Convert train-id map to color map for visualization.

    Args:
        train_ids: (H, W) with values in [0, 18] or 255(ignore).
        bgr_output: If True, output BGR-style colors to match your dataset visuals.
        ignore_color: RGB color for ignore regions.
    """
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
