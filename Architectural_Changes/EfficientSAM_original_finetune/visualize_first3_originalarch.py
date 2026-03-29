from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw

from cityscapes_semantic_utils import (
    build_prompt_targets_from_train_ids,
    decode_cityscapes_like_label_to_train_ids,
    encode_train_ids_to_color,
    pair_image_label_files,
)
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.expand_dims(array, axis=-1)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def select_best_mask_per_query(pred_masks: torch.Tensor, pred_iou: torch.Tensor) -> torch.Tensor:
    _, _, _, h, w = pred_masks.shape
    best_idx = torch.argmax(pred_iou, dim=-1, keepdim=True)
    gather_index = best_idx[..., None, None].expand(-1, -1, 1, h, w)
    return torch.gather(pred_masks, dim=2, index=gather_index).squeeze(2)


def build_semantic_prediction_from_queries(
    query_mask_logits: torch.Tensor,
    class_ids: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
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


def render_seg_overlay(base_image: Image.Image, seg_rgb: np.ndarray, alpha: float = 0.60) -> Image.Image:
    seg = Image.fromarray(seg_rgb, mode="RGB")
    return Image.blend(base_image.convert("RGB"), seg, alpha=alpha)


def make_panel(rows: List[Tuple[Image.Image, Image.Image, Image.Image, str]], output_path: Path) -> None:
    if not rows:
        raise RuntimeError("No rows provided to panel renderer.")

    tile_w = 256
    tile_h = 96
    title_h = 22
    canvas = Image.new("RGB", (tile_w * 3, len(rows) * (tile_h + title_h)), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    headers = [
        "Input Image",
        "Sobel filter used",
        "Original Arch Pred Mask",
    ]

    for row_idx, (inp, middle_img, pred_seg, rel_key) in enumerate(rows):
        y0 = row_idx * (tile_h + title_h)
        tiles = [inp, middle_img, pred_seg]
        for col_idx, tile in enumerate(tiles):
            x0 = col_idx * tile_w
            resized = tile.resize((tile_w, tile_h), resample=Image.BILINEAR)
            canvas.paste(resized, (x0, y0 + title_h))
            draw.rectangle((x0, y0 + title_h, x0 + tile_w - 1, y0 + title_h + tile_h - 1), outline=(180, 180, 180))
            draw.text((x0 + 8, y0 + 4), headers[col_idx], fill=(0, 0, 0))
        draw.text((8, y0 + title_h + 6), f"{rel_key}", fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    root = Path(__file__).resolve().parent
    dataset_root = Path("/home/aaditya/sandy/dataset")
    train_img_dir = dataset_root / "train" / "img"
    train_label_dir = dataset_root / "train" / "label"
    out_path = Path("/home/aaditya/sandy/compare_outputs/first3_train_originalarch_visuals.png")

    pairs = pair_image_label_files(train_img_dir, train_label_dir)
    if len(pairs) < 3:
        raise RuntimeError(f"Expected >=3 train pairs, found {len(pairs)}")
    first3 = pairs[:3]

    device = torch.device("cpu")
    torch.set_grad_enabled(False)

    cwd = Path.cwd()
    try:
        # The model builder uses a relative checkpoint path: weights/efficient_sam_vitt.pt
        # Ensure we run from project root so checkpoint resolution works.
        os.chdir(root)
        model = build_efficient_sam_vitt().to(device).eval()
    finally:
        os.chdir(cwd)

    rows = []
    for image_path, label_path, rel_key in first3:
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        image_tensor = pil_to_tensor(image).unsqueeze(0).to(device)
        label_rgb = np.asarray(label, dtype=np.uint8)
        train_ids = decode_cityscapes_like_label_to_train_ids(
            label_rgb,
            assume_bgr=True,
            max_color_distance=55.0,
        )

        points, point_labels, _target_masks, _valid_masks, class_ids = build_prompt_targets_from_train_ids(
            train_ids=train_ids,
            max_queries=19,
            num_classes=19,
            ignore_index=255,
            min_class_pixels=16,
            randomize=False,
            rng=random.Random(42),
        )
        points_t = torch.from_numpy(points).unsqueeze(0).to(device)
        point_labels_t = torch.from_numpy(point_labels.astype(np.int64)).unsqueeze(0).to(device)
        class_ids_t = torch.from_numpy(class_ids.astype(np.int64)).unsqueeze(0).to(device)

        pred_masks, pred_iou = model(
            image_tensor,
            points_t,
            point_labels_t,
            scale_to_original_image_size=True,
        )
        query_logits = select_best_mask_per_query(pred_masks, pred_iou)
        pred_labels = build_semantic_prediction_from_queries(
            query_logits,
            class_ids_t,
            num_classes=19,
        )[0]

        gt_seg_rgb = encode_train_ids_to_color(train_ids, bgr_output=False)
        pred_seg_rgb = encode_train_ids_to_color(pred_labels.cpu().numpy().astype(np.uint8), bgr_output=False)

        middle_overlay = render_seg_overlay(image, gt_seg_rgb, alpha=0.60)
        pred_overlay = render_seg_overlay(image, pred_seg_rgb, alpha=0.60)

        rows.append(
            (
                image,
                middle_overlay,
                pred_overlay,
                rel_key,
            )
        )

    make_panel(rows, out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
