"""
prepare_data.py — Slice large iSAID aerial images into 1024×1024 patches.

Reads raw images + COCO-format annotations, generates patches with updated
bounding boxes / segmentation polygons, and filters out background-only patches.

Usage:
    python prepare_data.py --data_root dataset --output_root dataset_patched \
                           --patch_size 1024 --stride 512 --bg_keep_ratio 0.1
"""

import argparse
import json
import os
import random
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def clip_bbox(bbox, patch_x, patch_y, patch_w, patch_h):
    """Shift and clip a COCO bbox [x, y, w, h] into patch coordinates.
    Returns None if the clipped box has zero or negative area."""
    x, y, w, h = bbox
    x1 = max(x - patch_x, 0)
    y1 = max(y - patch_y, 0)
    x2 = min(x + w - patch_x, patch_w)
    y2 = min(y + h - patch_y, patch_h)
    new_w = x2 - x1
    new_h = y2 - y1
    if new_w <= 2 or new_h <= 2:
        return None
    return [float(x1), float(y1), float(new_w), float(new_h)]


def clip_polygon(segmentation, patch_x, patch_y, patch_w, patch_h):
    """Shift polygon points into patch coordinates and clip to patch bounds.
    Returns the shifted polygon list or None if fewer than 3 points remain."""
    new_segs = []
    for poly in segmentation:
        coords = np.array(poly).reshape(-1, 2)
        coords[:, 0] -= patch_x
        coords[:, 1] -= patch_y
        # Clip to patch bounds
        coords[:, 0] = np.clip(coords[:, 0], 0, patch_w)
        coords[:, 1] = np.clip(coords[:, 1], 0, patch_h)
        # Remove degenerate polygons
        if len(coords) >= 3:
            new_segs.append(coords.flatten().tolist())
    return new_segs if new_segs else None


def bbox_from_polygon(segmentation):
    """Compute an axis-aligned bounding box [x, y, w, h] from polygon coords."""
    all_pts = []
    for poly in segmentation:
        coords = np.array(poly).reshape(-1, 2)
        all_pts.append(coords)
    pts = np.concatenate(all_pts, axis=0)
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_split(data_root, output_root, split, patch_size, stride, bg_keep_ratio):
    """Process one split (train / val)."""
    img_dir = os.path.join(data_root, split, "Images")
    mask_dir = os.path.join(data_root, split, "Instance_masks")
    ann_file = os.path.join(data_root, split, "Annotations",
                            f"iSAID_{split}.json")

    out_img_dir = os.path.join(output_root, split, "Images")
    out_mask_dir = os.path.join(output_root, split, "Instance_masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    # Load COCO annotations
    print(f"[{split}] Loading annotations from {ann_file} ...")
    with open(ann_file, "r") as f:
        coco = json.load(f)

    # Build look-ups
    images_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    patch_id = 0
    ann_id = 0
    skipped_bg = 0
    kept_bg = 0

    for img_info in tqdm(coco["images"], desc=f"[{split}] Slicing images"):
        fname = img_info["file_name"]
        img_path = os.path.join(img_dir, fname)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Also try to load instance mask
        stem = Path(fname).stem  # e.g. P0000
        mask_name = f"{stem}_instance_id_RGB.png"
        mask_path = os.path.join(mask_dir, mask_name)
        inst_mask = cv2.imread(mask_path) if os.path.exists(mask_path) else None

        img_anns = anns_by_image.get(img_info["id"], [])

        # Sliding window
        for y0 in range(0, img_h, stride):
            for x0 in range(0, img_w, stride):
                x1 = min(x0 + patch_size, img_w)
                y1 = min(y0 + patch_size, img_h)
                pw = x1 - x0
                ph = y1 - y0

                # Skip very small edge patches
                if pw < patch_size // 2 or ph < patch_size // 2:
                    continue

                # Collect annotations that fall in this patch
                patch_anns = []
                for ann in img_anns:
                    # Check if annotation overlaps this patch
                    if "segmentation" in ann and ann["segmentation"]:
                        new_seg = clip_polygon(ann["segmentation"],
                                               x0, y0, pw, ph)
                        if new_seg is None:
                            continue
                        new_bbox = bbox_from_polygon(new_seg)
                    elif "bbox" in ann:
                        new_bbox = clip_bbox(ann["bbox"], x0, y0, pw, ph)
                        if new_bbox is None:
                            continue
                        new_seg = None
                    else:
                        continue

                    # Skip tiny remnant annotations
                    if new_bbox[2] < 3 or new_bbox[3] < 3:
                        continue

                    new_ann = deepcopy(ann)
                    new_ann["bbox"] = new_bbox
                    new_ann["area"] = float(new_bbox[2] * new_bbox[3])
                    if new_seg is not None:
                        new_ann["segmentation"] = new_seg
                    patch_anns.append(new_ann)

                # Background filtering
                if len(patch_anns) == 0:
                    if random.random() > bg_keep_ratio:
                        skipped_bg += 1
                        continue
                    else:
                        kept_bg += 1

                # Extract and save patch
                patch = img[y0:y1, x0:x1]
                # Pad to full patch_size if on an edge
                if pw < patch_size or ph < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:ph, :pw] = patch
                    patch = padded

                patch_fname = f"{stem}_patch_{patch_id:06d}.png"
                cv2.imwrite(os.path.join(out_img_dir, patch_fname), patch)

                # Save instance mask patch
                if inst_mask is not None:
                    mask_crop = inst_mask[y0:y1, x0:x1]
                    if pw < patch_size or ph < patch_size:
                        padded_m = np.zeros((patch_size, patch_size, 3),
                                            dtype=np.uint8)
                        padded_m[:ph, :pw] = mask_crop
                        mask_crop = padded_m
                    mask_patch_fname = f"{stem}_patch_{patch_id:06d}_inst.png"
                    cv2.imwrite(os.path.join(out_mask_dir, mask_patch_fname),
                                mask_crop)

                # Register in COCO structures
                new_img = {
                    "id": patch_id,
                    "file_name": patch_fname,
                    "width": patch_size,
                    "height": patch_size,
                    "source_file": fname,
                    "patch_offset": [x0, y0],
                }
                new_images.append(new_img)

                for pa in patch_anns:
                    pa["id"] = ann_id
                    pa["image_id"] = patch_id
                    new_annotations.append(pa)
                    ann_id += 1

                patch_id += 1

    # Build output COCO JSON
    out_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }
    out_ann_path = os.path.join(output_root, split, "annotations.json")
    print(f"[{split}] Writing {len(new_images)} patches, "
          f"{len(new_annotations)} annotations to {out_ann_path}")
    print(f"[{split}] Skipped {skipped_bg} background patches, "
          f"kept {kept_bg} background patches")
    with open(out_ann_path, "w") as f:
        json.dump(out_coco, f)


def main():
    parser = argparse.ArgumentParser(
        description="Slice iSAID images into patches with updated annotations")
    parser.add_argument("--data_root", type=str, default="dataset",
                        help="Root of the raw iSAID dataset")
    parser.add_argument("--output_root", type=str, default="dataset_patched",
                        help="Output directory for patched data")
    parser.add_argument("--patch_size", type=int, default=1024,
                        help="Patch size in pixels (default: 1024)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride (default: 512)")
    parser.add_argument("--bg_keep_ratio", type=float, default=0.1,
                        help="Fraction of background-only patches to keep "
                             "(default: 0.1, i.e. drop 90%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    for split in ["train", "val"]:
        process_split(args.data_root, args.output_root, split,
                      args.patch_size, args.stride, args.bg_keep_ratio)

    print("\n✅ Data preparation complete!")


if __name__ == "__main__":
    main()
