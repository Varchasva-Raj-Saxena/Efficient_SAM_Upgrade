"""
dataset.py — PyTorch Dataset for patched iSAID data.

Loads image patches and instance masks, returns image tensors, binary GT masks,
and noisy bounding-box prompts formatted for EfficientSAM.

Memory-optimised: stores only lightweight numeric arrays instead of full
annotation dicts, avoiding OOM when Windows multiprocessing pickles workers.

Usage:
    from dataset import get_dataloaders
    train_loader, val_loader = get_dataloaders("dataset_patched", batch_size=4)
"""

import json
import os
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants — ImageNet normalization used by EfficientSAM
# ---------------------------------------------------------------------------
PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 1024          # EfficientSAM native resolution
MASK_OUT_SIZE = 256      # Decoder output resolution


class iSAIDPatchDataset(Dataset):
    """
    Dataset for patched iSAID images.

    Each sample returns:
        image       : float32 tensor [3, 1024, 1024]  (normalised)
        gt_mask     : float32 tensor [1, 256, 256]     (binary)
        bbox_points : float32 tensor [1, 2, 2]         (noisy top-left & bottom-right)
        bbox_labels : int64   tensor [1, 2]            ([2, 3] = SAM bbox labels)
    """

    def __init__(self, root: str, split: str = "train",
                 jitter_range: tuple = (5, 15)):
        super().__init__()
        self.root = root
        self.split = split
        self.jitter_range = jitter_range

        self.img_dir = os.path.join(root, split, "Images")
        self.mask_dir = os.path.join(root, split, "Instance_masks")
        ann_path = os.path.join(root, split, "annotations.json")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # -----------------------------------------------------------
        # Memory-efficient storage:
        #   image_fnames : list[str]  — one per image id
        #   image_id_map : dict[int → int]  — coco id → list index
        #   sample_image_ids : int32 array  — image index per sample
        #   sample_bboxes   : float32 array [N, 4] — (x, y, w, h)
        # We do NOT keep the full annotation dicts in memory.
        # -----------------------------------------------------------
        self.image_fnames = []
        image_id_to_idx = {}
        for img in coco["images"]:
            image_id_to_idx[img["id"]] = len(self.image_fnames)
            self.image_fnames.append(img["file_name"])

        img_ids = []
        bboxes = []
        for ann in coco["annotations"]:
            bbox = ann.get("bbox")
            if bbox is None or bbox[2] <= 0 or bbox[3] <= 0:
                continue
            idx = image_id_to_idx.get(ann["image_id"])
            if idx is None:
                continue
            img_ids.append(idx)
            bboxes.append(bbox)

        self.sample_image_ids = np.array(img_ids, dtype=np.int32)
        self.sample_bboxes = np.array(bboxes, dtype=np.float32)

        print(f"[{split}] Loaded {len(self.sample_image_ids)} instance "
              f"samples across {len(self.image_fnames)} patches  "
              f"(dataset obj ~{sys.getsizeof(self.sample_bboxes) / 1e6:.1f} MB)")

    def __len__(self):
        return len(self.sample_image_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Load an image as RGB float32 in [0, 1]."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.float32) / 255.0

    @staticmethod
    def _extract_binary_mask(inst_mask_rgb: np.ndarray,
                             bbox) -> np.ndarray:
        """
        Extract a binary mask for one instance from the RGB instance-id mask.

        iSAID instance masks encode each instance as a unique RGB colour.
        We sample the colour at the bbox centre, then threshold.
        """
        h, w = inst_mask_rgb.shape[:2]
        bx, by, bw, bh = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        cx = int(min(max(bx + bw / 2, 0), w - 1))
        cy = int(min(max(by + bh / 2, 0), h - 1))
        target_colour = inst_mask_rgb[cy, cx].copy()

        # If centre is background, sample random points in the bbox
        if np.all(target_colour == 0):
            for _ in range(20):
                rx = int(random.uniform(bx, min(bx + bw, w - 1)))
                ry = int(random.uniform(by, min(by + bh, h - 1)))
                c = inst_mask_rgb[ry, rx]
                if not np.all(c == 0):
                    target_colour = c.copy()
                    break

        mask = np.all(inst_mask_rgb == target_colour[None, None, :], axis=2)
        return mask.astype(np.float32)

    def _add_bbox_jitter(self, bbox, img_w: int, img_h: int):
        """Add random jitter to bounding box coordinates."""
        x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        jitter = random.randint(*self.jitter_range)
        sign = lambda: random.choice([-1, 1])
        x1 = max(0, x + sign() * jitter)
        y1 = max(0, y + sign() * jitter)
        x2 = min(img_w, x + w + sign() * jitter)
        y2 = min(img_h, y + h + sign() * jitter)
        if x2 <= x1:
            x2 = min(x1 + 2, img_w)
        if y2 <= y1:
            y2 = min(y1 + 2, img_h)
        return x1, y1, x2, y2

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        img_idx = int(self.sample_image_ids[idx])
        bbox = self.sample_bboxes[idx]          # numpy float32 [4]
        fname = self.image_fnames[img_idx]

        # --- Load image ---
        img_path = os.path.join(self.img_dir, fname)
        img = self._load_image(img_path)

        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        img = (img - PIXEL_MEAN) / PIXEL_STD
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # --- Load GT mask ---
        stem = fname.replace(".png", "")
        mask_fname = f"{stem}_inst.png"
        mask_path = os.path.join(self.mask_dir, mask_fname)

        if os.path.exists(mask_path):
            inst_mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            inst_mask = cv2.cvtColor(inst_mask, cv2.COLOR_BGR2RGB)
            binary_mask = self._extract_binary_mask(inst_mask, bbox)
        else:
            # Fallback: empty mask (polygon data not stored for memory)
            binary_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        mask_resized = cv2.resize(binary_mask, (MASK_OUT_SIZE, MASK_OUT_SIZE),
                                  interpolation=cv2.INTER_NEAREST)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float()

        # --- BBox prompt (noisy) ---
        x1, y1, x2, y2 = self._add_bbox_jitter(bbox, IMG_SIZE, IMG_SIZE)
        bbox_points = torch.tensor([[[x1, y1], [x2, y2]]], dtype=torch.float32)
        bbox_labels = torch.tensor([[2, 3]], dtype=torch.int64)

        return img_tensor, mask_tensor, bbox_points, bbox_labels


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(data_root: str = "dataset_patched",
                    batch_size: int = 4,
                    num_workers: int = 0):
    """
    Create train and validation DataLoaders.

    Default num_workers=0 (main process) to avoid Windows multiprocessing
    MemoryError. Increase to 2 only if you have ample RAM (>32 GB).
    """
    train_ds = iSAIDPatchDataset(data_root, split="train")
    val_ds = iSAIDPatchDataset(data_root, split="val")

    loader_kwargs = dict(
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
