"""
Create a small labeled test split from an existing patched split.

This script samples a subset of patched images (default: 200) from
`dataset_patched/val` and writes:

    test_dataset_patched/
      test/
        Images/
        Instance_masks/
        annotations.json

The output keeps COCO-style annotations and category metadata, with image and
annotation ids re-indexed for a clean standalone split.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


def build_small_test_split(
    source_root: str,
    output_root: str,
    num_images: int,
    seed: int,
) -> None:
    src_split_dir = Path(source_root)
    src_img_dir = src_split_dir / "Images"
    src_mask_dir = src_split_dir / "Instance_masks"
    src_ann_path = src_split_dir / "annotations.json"

    if not src_ann_path.is_file():
        raise FileNotFoundError(f"Missing source annotations: {src_ann_path}")

    with open(src_ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if not images:
        raise RuntimeError("No images found in source annotations.")

    rng = random.Random(seed)
    n = min(num_images, len(images))
    selected_images = rng.sample(images, n)
    selected_image_ids = {img["id"] for img in selected_images}

    selected_anns = [a for a in anns if a.get("image_id") in selected_image_ids]

    out_split_dir = Path(output_root) / "test"
    out_img_dir = out_split_dir / "Images"
    out_mask_dir = out_split_dir / "Instance_masks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    # Re-index image/annotation ids so the output is self-contained and compact.
    old_to_new_img_id = {}
    remapped_images = []
    for new_img_id, img in enumerate(selected_images):
        old_to_new_img_id[img["id"]] = new_img_id
        new_img = dict(img)
        new_img["id"] = new_img_id
        remapped_images.append(new_img)

        fname = img["file_name"]
        src_img = src_img_dir / fname
        if not src_img.is_file():
            raise FileNotFoundError(f"Missing image file: {src_img}")
        shutil.copy2(src_img, out_img_dir / fname)

        stem = fname.replace(".png", "")
        mask_fname = f"{stem}_inst.png"
        src_mask = src_mask_dir / mask_fname
        if src_mask.is_file():
            shutil.copy2(src_mask, out_mask_dir / mask_fname)

    remapped_anns = []
    for new_ann_id, ann in enumerate(selected_anns):
        new_ann = dict(ann)
        new_ann["id"] = new_ann_id
        new_ann["image_id"] = old_to_new_img_id[ann["image_id"]]
        remapped_anns.append(new_ann)

    out_coco = {
        "images": remapped_images,
        "annotations": remapped_anns,
        "categories": categories,
    }
    out_ann_path = out_split_dir / "annotations.json"
    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(out_coco, f)

    print(
        f"Created {out_ann_path}\n"
        f"  images: {len(remapped_images)}\n"
        f"  annotations: {len(remapped_anns)}\n"
        f"  categories: {len(categories)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small patched test set from patched val/train split."
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default="dataset_patched/val",
        help="Path to source patched split containing Images/ Instance_masks/ annotations.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="test_dataset_patched",
        help="Output root where test/ will be created",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=200,
        help="Number of images to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    build_small_test_split(
        source_root=args.source_root,
        output_root=args.output_root,
        num_images=args.num_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
