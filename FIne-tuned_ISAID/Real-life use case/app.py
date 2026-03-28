# -*- coding: utf-8 -*-
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from torch.amp import autocast


# Keep Gradio temporary files local to this project directory.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(APP_DIR, "gradio_tmp")
os.makedirs(TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = TEMP_DIR

PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import IMG_SIZE, PIXEL_MEAN, PIXEL_STD  # noqa: E402
from model_setup import freeze_model, inject_lora, load_efficient_sam  # noqa: E402


CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_mask_decoder.pth")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results_comparison")
INFERENCE_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "inference_outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DEVICE = "cuda" if DEVICE.startswith("cuda") else "cpu"
MAX_POINTS = 6
BASELINE_MODEL_TYPE = "vits"
BASELINE_MODEL_LABEL = "EfficientSAM ViT-S"
ACCENT_COLORS = {
    "base": (59, 130, 246),
    "ft": (16, 185, 129),
    "warn": (245, 158, 11),
    "muted": (148, 163, 184),
}


def _empty_prompt_state() -> Dict[str, List]:
    return {"box_points": [], "point_prompts": []}


def _load_models():
    print(f"Loading EfficientSAM models on {DEVICE}...")
    base_model = load_efficient_sam(BASELINE_MODEL_TYPE, device=DEVICE)
    base_model.eval()

    finetuned_model = None
    finetuned_message = (
        "Fine-tuned checkpoint not found. The app will still run with the base model."
    )

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model_type = ckpt.get("model_type", "vits")
        finetuned_model = load_efficient_sam(model_type, device=DEVICE)
        freeze_model(finetuned_model)

        use_lora = bool(ckpt.get("use_lora", False))
        if use_lora:
            inject_lora(
                finetuned_model,
                rank=int(ckpt.get("lora_rank", 4)),
                alpha=float(ckpt.get("lora_alpha", 1.0)),
            )

        finetuned_model.mask_decoder.load_state_dict(ckpt["mask_decoder"], strict=True)

        if use_lora and "lora_params" in ckpt:
            param_dict = dict(finetuned_model.image_encoder.named_parameters())
            for name, value in ckpt["lora_params"].items():
                if name in param_dict:
                    param_dict[name].data.copy_(value.to(DEVICE))

        finetuned_model = finetuned_model.to(DEVICE)
        finetuned_model.eval()
        finetuned_message = (
            f"Fine-tuned checkpoint loaded from {CHECKPOINT_PATH} "
            f"(checkpoint backbone: {model_type})."
        )

    return base_model, finetuned_model, finetuned_message


BASE_MODEL, FINETUNED_MODEL, MODEL_STATUS = _load_models()


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("No image provided.")
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _image_to_native_model_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def _image_to_legacy_finetuned_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img = resized.astype(np.float32) / 255.0
    img = (img - PIXEL_MEAN) / PIXEL_STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)


def _prepare_image_state(image: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Dict]:
    if image is None:
        return None, {"image_rgb": None, "base_embedding": None, "ft_embedding": None}

    image_rgb = _ensure_rgb(image).astype(np.uint8)
    native_tensor = _image_to_native_model_tensor(image_rgb).to(DEVICE)
    legacy_ft_tensor = _image_to_legacy_finetuned_tensor(image_rgb).to(DEVICE)

    with torch.no_grad(), autocast(device_type=AMP_DEVICE):
        base_embedding = BASE_MODEL.get_image_embeddings(native_tensor)
        ft_embedding = (
            FINETUNED_MODEL.get_image_embeddings(legacy_ft_tensor)
            if FINETUNED_MODEL is not None
            else None
        )

    return image_rgb, {
        "image_rgb": image_rgb,
        "base_embedding": base_embedding,
        "ft_embedding": ft_embedding,
    }


def _make_status_text(prompt_mode: str, prompt_state: Dict) -> str:
    if prompt_mode == "Bounding Box":
        box_points = prompt_state["box_points"]
        if not box_points:
            return "Click once for the first corner of the box."
        if len(box_points) == 1:
            x, y = box_points[0]
            return f"First box corner fixed at ({x}, {y}). Click the opposite corner."
        x1, y1 = box_points[0]
        x2, y2 = box_points[1]
        return f"Bounding box ready from ({x1}, {y1}) to ({x2}, {y2})."

    point_prompts = prompt_state["point_prompts"]
    if not point_prompts:
        return "Click one or more object points on the image."
    return f"{len(point_prompts)} point prompt(s) collected. You can add up to {MAX_POINTS}."


def _draw_prompt_preview(image_rgb: np.ndarray, prompt_mode: str, prompt_state: Dict) -> np.ndarray:
    preview = image_rgb.copy()

    if prompt_mode == "Bounding Box":
        box_points = prompt_state["box_points"]
        if len(box_points) >= 1:
            x, y = box_points[0]
            cv2.circle(preview, (x, y), 8, (255, 196, 0), -1)
        if len(box_points) >= 2:
            x1, y1 = box_points[0]
            x2, y2 = box_points[1]
            cv2.rectangle(
                preview,
                (min(x1, x2), min(y1, y2)),
                (max(x1, x2), max(y1, y2)),
                (255, 196, 0),
                3,
            )
    else:
        for idx, (x, y) in enumerate(prompt_state["point_prompts"], start=1):
            cv2.circle(preview, (x, y), 7, (34, 197, 94), -1)
            cv2.circle(preview, (x, y), 12, (255, 255, 255), 2)
            cv2.putText(
                preview,
                str(idx),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                lineType=cv2.LINE_AA,
            )

    return preview


def _build_prompt_tensors(
    image_rgb: np.ndarray, prompt_mode: str, prompt_state: Dict
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    h, w = image_rgb.shape[:2]

    if prompt_mode == "Bounding Box":
        if len(prompt_state["box_points"]) < 2:
            raise ValueError("Please click two corners to define a bounding box.")
        (x1, y1), (x2, y2) = prompt_state["box_points"][:2]
        points = [[[float(min(x1, x2)), float(min(y1, y2))], [float(max(x1, x2)), float(max(y1, y2))]]]
        labels = [[2, 3]]
        prompt_summary = (
            f"Bounding box: ({min(x1, x2)}, {min(y1, y2)}) to ({max(x1, x2)}, {max(y1, y2)})"
        )
    else:
        prompts = prompt_state["point_prompts"][:MAX_POINTS]
        if not prompts:
            raise ValueError("Please click at least one point on the target object.")
        points = [[list(map(float, prompt)) for prompt in prompts]]
        labels = [[1] * len(prompts)]
        prompt_summary = f"Point prompts: {len(prompts)} selected"

    return (
        torch.tensor(points, dtype=torch.float32, device=DEVICE).unsqueeze(0),
        torch.tensor(labels, dtype=torch.int64, device=DEVICE).unsqueeze(0),
        prompt_summary + f" on an image of size {w}x{h}",
    )


def _predict_mask(
    model,
    image_embedding,
    image_rgb: np.ndarray,
    points: torch.Tensor,
    labels: torch.Tensor,
    use_multimask: bool,
):
    with torch.no_grad(), autocast(device_type=AMP_DEVICE):
        masks, iou_scores = model.predict_masks(
            image_embeddings=image_embedding,
            batched_points=points,
            batched_point_labels=labels,
            multimask_output=use_multimask,
            input_h=image_rgb.shape[0],
            input_w=image_rgb.shape[1],
            output_h=image_rgb.shape[0],
            output_w=image_rgb.shape[1],
        )

    if use_multimask:
        candidate_scores = iou_scores[0, 0]
        best_idx = int(torch.argmax(candidate_scores).detach().cpu().item())
        mask_logits = masks[0, 0, best_idx]
        score = float(candidate_scores[best_idx].detach().cpu().item())
    else:
        mask_logits = masks[0, 0, 0]
        score = float(iou_scores[0, 0, 0].detach().cpu().item())

    mask = (mask_logits > 0.0).detach().cpu().numpy().astype(np.uint8)
    return mask, score


def _masked_pixels_only(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    panel = np.zeros_like(image_rgb)
    panel[mask > 0] = image_rgb[mask > 0]
    return panel


def _mask_panel(
    image_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    score: Optional[float],
    prompt_mode: str,
    prompt_state: Dict,
    color: Tuple[int, int, int],
    title: str,
) -> Optional[np.ndarray]:
    if mask is None:
        return None

    panel = _masked_pixels_only(image_rgb, mask)
    panel = _draw_prompt_preview(panel, prompt_mode, prompt_state)
    coverage = 100.0 * float(mask.sum()) / float(mask.size)
    cv2.putText(
        panel,
        title,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Mask coverage: {coverage:.2f}%",
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    if score is not None:
        cv2.putText(
            panel,
            f"Predicted quality: {score:.3f}",
            (18, 96),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return panel


def _difference_panel(image_rgb: np.ndarray, base_mask: np.ndarray, ft_mask: np.ndarray) -> np.ndarray:
    panel = image_rgb.copy()
    both = np.logical_and(base_mask == 1, ft_mask == 1)
    only_base = np.logical_and(base_mask == 1, ft_mask == 0)
    only_ft = np.logical_and(base_mask == 0, ft_mask == 1)

    color_layer = np.zeros_like(panel)
    color_layer[both] = (255, 255, 255)
    color_layer[only_base] = (239, 68, 68)
    color_layer[only_ft] = (34, 197, 94)
    panel = cv2.addWeighted(panel, 1.0, color_layer, 0.45, 0.0)

    cv2.putText(
        panel,
        "Difference Map",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Red: base only | Green: fine-tuned only | White: agreement",
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return panel


def _comparison_rows(
    prompt_mode: str,
    prompt_state: Dict,
    base_mask: np.ndarray,
    base_score: float,
    ft_mask: Optional[np.ndarray],
    ft_score: Optional[float],
) -> List[List[str]]:
    prompt_count = (
        len(prompt_state["box_points"]) if prompt_mode == "Bounding Box" else len(prompt_state["point_prompts"])
    )
    base_area = int(base_mask.sum())
    total_pixels = int(base_mask.size)
    rows = [
        ["Prompt type", prompt_mode],
        ["Prompt count", str(prompt_count)],
        ["Base mask pixels", f"{base_area:,} / {total_pixels:,}"],
        ["Base predicted quality", f"{base_score:.4f}"],
    ]

    if ft_mask is not None and ft_score is not None:
        ft_area = int(ft_mask.sum())
        intersection = int(np.logical_and(base_mask == 1, ft_mask == 1).sum())
        union = int(np.logical_or(base_mask == 1, ft_mask == 1).sum())
        diff_pixels = int(np.not_equal(base_mask, ft_mask).sum())
        agreement = 1.0 - (diff_pixels / float(total_pixels))
        overlap_iou = (intersection / union) if union > 0 else 1.0

        rows.extend(
            [
                ["Fine-tuned mask pixels", f"{ft_area:,} / {total_pixels:,}"],
                ["Fine-tuned predicted quality", f"{ft_score:.4f}"],
                ["Mask overlap IoU", f"{overlap_iou:.4f}"],
                ["Pixel agreement", f"{agreement:.4f}"],
                ["Changed pixels", f"{diff_pixels:,}"],
            ]
        )
    else:
        rows.append(["Fine-tuned model", "Checkpoint unavailable"])

    return rows


def _metric_lookup(rows: List[List[str]]) -> Dict[str, str]:
    return {row[0]: row[1] for row in rows if len(row) >= 2}


def _extract_numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group()) if match else None


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _summary_cards_html(title: str, subtitle: str, cards: List[Tuple[str, str, str]]) -> str:
    card_html = []
    for label, value, tone in cards:
        card_html.append(
            f"""
            <div class="metric-card metric-card-{tone}">
              <div class="metric-label">{_html_escape(label)}</div>
              <div class="metric-value">{_html_escape(value)}</div>
            </div>
            """
        )
    return f"""
    <div class="summary-shell">
      <div class="summary-header">
        <div>
          <div class="eyebrow">Performance Snapshot</div>
          <h3>{_html_escape(title)}</h3>
          <p>{_html_escape(subtitle)}</p>
        </div>
      </div>
      <div class="metric-grid">
        {''.join(card_html)}
      </div>
    </div>
    """


def _blank_chart(width: int = 1200, height: int = 680) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array([15, 23, 42], dtype=np.float32)
    bottom = np.array([28, 62, 95], dtype=np.float32)
    for y in range(height):
        alpha = y / max(height - 1, 1)
        color = ((1.0 - alpha) * top + alpha * bottom).astype(np.uint8)
        canvas[y, :] = color
    return canvas


def _put_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def _grouped_bar_chart(
    title: str,
    categories: List[str],
    series: List[Tuple[str, List[Optional[float]], Tuple[int, int, int]]],
    subtitle: str,
) -> np.ndarray:
    canvas = _blank_chart()
    h, w = canvas.shape[:2]
    left, right, top, bottom = 120, 70, 150, 110
    chart_w = w - left - right
    chart_h = h - top - bottom

    cv2.rectangle(canvas, (48, 36), (w - 48, h - 36), (255, 255, 255), 1)
    _put_text(canvas, title, 72, 86, 1.15, (255, 255, 255), 2)
    _put_text(canvas, subtitle, 72, 118, 0.62, (200, 226, 239), 1)

    max_val = 0.0
    for _, values, _ in series:
        for value in values:
            if value is not None:
                max_val = max(max_val, value)
    max_val = max(1.0, max_val * 1.15)

    for tick in range(6):
        y = top + int(chart_h * tick / 5.0)
        value = max_val * (1.0 - tick / 5.0)
        cv2.line(canvas, (left, y), (w - right, y), (70, 102, 130), 1)
        _put_text(canvas, f"{value:.2f}", 44, y + 5, 0.5, (196, 220, 233), 1)

    group_w = chart_w / max(len(categories), 1)
    bar_w = max(int(group_w / (len(series) + 1.5)), 24)
    for idx, category in enumerate(categories):
        group_x = left + int(idx * group_w)
        cat_center = group_x + int(group_w / 2)
        _put_text(canvas, category, max(cat_center - 45, 18), h - 48, 0.6, (235, 245, 255), 1)

        series_start = cat_center - int((len(series) * bar_w + (len(series) - 1) * 16) / 2)
        for series_idx, (_, values, color) in enumerate(series):
            value = values[idx]
            if value is None:
                continue
            bar_left = series_start + series_idx * (bar_w + 16)
            bar_height = int((value / max_val) * chart_h)
            bar_top = top + chart_h - bar_height
            cv2.rectangle(canvas, (bar_left, bar_top), (bar_left + bar_w, top + chart_h), color, -1)
            cv2.rectangle(canvas, (bar_left, bar_top), (bar_left + bar_w, top + chart_h), (255, 255, 255), 1)
            _put_text(canvas, f"{value:.3f}", bar_left - 4, max(bar_top - 10, top - 10), 0.46, (255, 255, 255), 1)

    legend_x = w - 340
    legend_y = 88
    for idx, (label, _, color) in enumerate(series):
        y = legend_y + idx * 30
        cv2.rectangle(canvas, (legend_x, y - 12), (legend_x + 18, y + 6), color, -1)
        _put_text(canvas, label, legend_x + 28, y + 2, 0.56, (240, 248, 255), 1)

    return canvas


def _live_comparison_summary(rows: List[List[str]], prompt_summary: str) -> str:
    metrics = _metric_lookup(rows)
    cards = [
        ("Prompt", metrics.get("Prompt type", "Unknown"), "neutral"),
        ("Prompt Count", metrics.get("Prompt count", "0"), "neutral"),
        ("Base Quality", metrics.get("Base predicted quality", "N/A"), "base"),
    ]
    if "Fine-tuned predicted quality" in metrics:
        cards.extend(
            [
                ("Fine-Tuned Quality", metrics["Fine-tuned predicted quality"], "ft"),
                ("Mask Overlap IoU", metrics.get("Mask overlap IoU", "N/A"), "ft"),
                ("Pixel Agreement", metrics.get("Pixel agreement", "N/A"), "neutral"),
            ]
        )
    else:
        cards.append(("Fine-Tuned Model", metrics.get("Fine-tuned model", "Unavailable"), "warn"))
    return _summary_cards_html("Live Segmentation Result", prompt_summary, cards)


def _live_metrics_chart(rows: List[List[str]]) -> Optional[np.ndarray]:
    metrics = _metric_lookup(rows)
    categories = ["Pred Quality", "Mask Coverage"]

    base_quality = _extract_numeric(metrics.get("Base predicted quality"))
    base_pixels = _extract_numeric(metrics.get("Base mask pixels"))
    total_pixels = None
    if metrics.get("Base mask pixels") and "/" in metrics["Base mask pixels"]:
        total_pixels = _extract_numeric(metrics["Base mask pixels"].split("/")[-1])
    base_coverage = (base_pixels / total_pixels) if base_pixels is not None and total_pixels else None

    series = [("Base", [base_quality, base_coverage], ACCENT_COLORS["base"])]

    ft_quality = _extract_numeric(metrics.get("Fine-tuned predicted quality"))
    ft_pixels = _extract_numeric(metrics.get("Fine-tuned mask pixels"))
    ft_coverage = (ft_pixels / total_pixels) if ft_pixels is not None and total_pixels else None
    if ft_quality is not None or ft_coverage is not None:
        series.append(("Fine-tuned", [ft_quality, ft_coverage], ACCENT_COLORS["ft"]))

    if len(series) == 1 and base_quality is None and base_coverage is None:
        return None

    return _grouped_bar_chart(
        "Interactive Result Metrics",
        categories,
        series,
        "Quality and coverage give a fast view of how each model responded to the same prompt.",
    )


def _live_delta_chart(rows: List[List[str]]) -> Optional[np.ndarray]:
    metrics = _metric_lookup(rows)
    overlap = _extract_numeric(metrics.get("Mask overlap IoU"))
    agreement = _extract_numeric(metrics.get("Pixel agreement"))
    changed = _extract_numeric(metrics.get("Changed pixels"))
    if overlap is None and agreement is None and changed is None:
        return None

    changed_ratio = None
    base_pixels_text = metrics.get("Base mask pixels")
    if changed is not None and base_pixels_text and "/" in base_pixels_text:
        total_pixels = _extract_numeric(base_pixels_text.split("/")[-1])
        if total_pixels:
            changed_ratio = changed / total_pixels

    return _grouped_bar_chart(
        "Model-to-Model Delta",
        ["Overlap IoU", "Agreement", "Changed Ratio"],
        [("Comparison", [overlap, agreement, changed_ratio], ACCENT_COLORS["warn"])],
        "This chart highlights where the fine-tuned output diverges from the baseline mask.",
    )


def _point_prompt_comparison_message() -> str:
    return (
        "Point-prompt mode is native for pretrained EfficientSAM, but your fine-tuned checkpoint "
        "was trained on bounding-box prompts only. Use Bounding Box mode for a fair comparison."
    )


def on_image_change(image: Optional[np.ndarray], prompt_mode: str):
    preview, image_state = _prepare_image_state(image)
    prompt_state = _empty_prompt_state()
    if preview is None:
        return None, prompt_state, image_state, "Upload a satellite or aerial image to begin."
    status = (
        "Image loaded and embeddings cached. "
        + _make_status_text(prompt_mode, prompt_state)
    )
    return preview, prompt_state, image_state, status


def on_mode_change(prompt_mode: str, image: Optional[np.ndarray]):
    prompt_state = _empty_prompt_state()
    if image is None:
        return None, prompt_state, _make_status_text(prompt_mode, prompt_state)
    image_rgb = _ensure_rgb(image).astype(np.uint8)
    preview = _draw_prompt_preview(image_rgb, prompt_mode, prompt_state)
    return preview, prompt_state, _make_status_text(prompt_mode, prompt_state)


def on_preview_click(
    evt: gr.SelectData,
    image: Optional[np.ndarray],
    prompt_mode: str,
    prompt_state: Dict,
):
    if image is None:
        return None, prompt_state, "Upload an image before adding prompts."

    x, y = int(evt.index[0]), int(evt.index[1])
    image_rgb = _ensure_rgb(image).astype(np.uint8)
    prompt_state = {
        "box_points": [list(p) for p in prompt_state.get("box_points", [])],
        "point_prompts": [list(p) for p in prompt_state.get("point_prompts", [])],
    }

    if prompt_mode == "Bounding Box":
        if len(prompt_state["box_points"]) >= 2:
            prompt_state["box_points"] = []
        prompt_state["box_points"].append([x, y])
    else:
        if len(prompt_state["point_prompts"]) >= MAX_POINTS:
            prompt_state["point_prompts"].pop(0)
        prompt_state["point_prompts"].append([x, y])

    preview = _draw_prompt_preview(image_rgb, prompt_mode, prompt_state)
    return preview, prompt_state, _make_status_text(prompt_mode, prompt_state)


def clear_prompts(image: Optional[np.ndarray], prompt_mode: str):
    prompt_state = _empty_prompt_state()
    if image is None:
        return None, prompt_state, "Upload an image to start adding prompts."
    image_rgb = _ensure_rgb(image).astype(np.uint8)
    preview = _draw_prompt_preview(image_rgb, prompt_mode, prompt_state)
    return preview, prompt_state, _make_status_text(prompt_mode, prompt_state)


def run_segmentation(prompt_mode: str, image_state: Dict, prompt_state: Dict):
    image_rgb = image_state.get("image_rgb") if image_state else None
    if image_rgb is None:
        return (
            "Upload an image first.",
            None,
            None,
            None,
            [["Fine-tuned model", "Unavailable"]],
            "No prompt prepared.",
            _summary_cards_html(
                "Live Segmentation Result",
                "Upload an image and add prompts to see a polished comparison summary here.",
                [
                    ("Prompt", "Waiting", "neutral"),
                    ("Base Quality", "N/A", "base"),
                    ("Fine-Tuned Quality", "N/A", "ft"),
                ],
            ),
            None,
            None,
        )

    try:
        points, labels, prompt_summary = _build_prompt_tensors(image_rgb, prompt_mode, prompt_state)
    except ValueError as exc:
        return (
            str(exc),
            None,
            None,
            None,
            [["Prompt", "Incomplete"]],
            "Prompt is incomplete.",
            _summary_cards_html(
                "Prompt Needs Attention",
                str(exc),
                [
                    ("Prompt", prompt_mode, "warn"),
                    ("Status", "Incomplete", "warn"),
                    ("Next Step", "Add the required clicks", "neutral"),
                ],
            ),
            None,
            None,
        )

    base_embedding = image_state["base_embedding"]
    base_mask, base_score = _predict_mask(
        BASE_MODEL, base_embedding, image_rgb, points, labels, use_multimask=True
    )
    base_panel = _mask_panel(
        image_rgb,
        base_mask,
        base_score,
        prompt_mode,
        prompt_state,
        color=(239, 68, 68),
        title="Base EfficientSAM",
    )

    finetuned_panel = None
    difference_panel = None
    ft_mask = None
    ft_score = None

    if prompt_mode == "Point Prompt":
        status = "Segmentation complete for the pretrained baseline. " + _point_prompt_comparison_message()
        comparison = _comparison_rows(prompt_mode, prompt_state, base_mask, base_score, None, None)
        comparison[-1] = ["Fine-tuned model", "Skipped in point-prompt mode"]
        return (
            status,
            base_panel,
            None,
            None,
            comparison,
            prompt_summary,
            _live_comparison_summary(comparison, prompt_summary),
            _live_metrics_chart(comparison),
            _live_delta_chart(comparison),
        )

    if FINETUNED_MODEL is not None and image_state.get("ft_embedding") is not None:
        ft_mask, ft_score = _predict_mask(
            FINETUNED_MODEL,
            image_state["ft_embedding"],
            image_rgb,
            points,
            labels,
            use_multimask=False,
        )
        finetuned_panel = _mask_panel(
            image_rgb,
            ft_mask,
            ft_score,
            prompt_mode,
            prompt_state,
            color=(34, 197, 94),
            title="Fine-tuned EfficientSAM",
        )
        difference_panel = _difference_panel(image_rgb, base_mask, ft_mask)

    comparison = _comparison_rows(prompt_mode, prompt_state, base_mask, base_score, ft_mask, ft_score)
    status = "Segmentation complete."
    if FINETUNED_MODEL is None:
        status += " Fine-tuned checkpoint is missing, so only the base result is shown."

    return (
        status,
        base_panel,
        finetuned_panel,
        difference_panel,
        comparison,
        prompt_summary,
        _live_comparison_summary(comparison, prompt_summary),
        _live_metrics_chart(comparison),
        _live_delta_chart(comparison),
    )


def _read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def _find_metrics_json_files() -> List[str]:
    return sorted(glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, "*", "metrics_summary.json")))


def _parse_output_metrics() -> Optional[Dict]:
    text = _read_text(os.path.join(RESULTS_DIR, "output.txt"))
    if not text:
        return None

    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    total_batches = None
    models: List[str] = []
    metrics: Dict[str, Dict[str, float]] = {}

    for line in lines:
        if line.startswith("Total batches evaluated:"):
            total_batches = line
        elif line.startswith("Metric") and "|" in line:
            parts = [part.strip() for part in line.split("|")]
            models = [part for part in parts[1:] if part]
        elif "|" in line and not set(line.replace("|", "").strip()) <= {"-"}:
            parts = [part.strip() for part in line.split("|")]
            if len(parts) >= 4 and parts[0] != "Metric":
                metric_name = parts[0]
                values = parts[1:1 + len(models)]
                metric_row = {}
                for model, value in zip(models, values):
                    numeric = _extract_numeric(value)
                    if numeric is not None:
                        metric_row[model] = numeric
                if metric_row:
                    metrics[metric_name] = metric_row

    if not models or not metrics:
        return None

    return {
        "total_batches": total_batches or "Saved benchmark summary available.",
        "models": models,
        "metrics": metrics,
    }


def _metrics_markdown() -> str:
    output_txt = os.path.join(RESULTS_DIR, "output.txt")
    text = _read_text(output_txt)
    if text:
        return f"```text\n{text}\n```"

    json_files = _find_metrics_json_files()
    if not json_files:
        return (
            "No saved benchmark metrics found yet.\n\n"
            "Run `compare_models.py`, `inference_base.py`, or `inference_finetuned.py` "
            "to generate dataset-level comparison outputs."
        )

    blocks = []
    for path in json_files:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        blocks.append(
            "\n".join(
                [
                    f"Run: {data.get('run_name', os.path.basename(os.path.dirname(path)))}",
                    f"Split: {data.get('split', 'unknown')}",
                    f"Instances evaluated: {data.get('num_annotations_eval', 0)}",
                    f"Mean IoU: {data.get('macro_over_instances', {}).get('mean_iou', 0.0):.4f}",
                    f"Mean Dice: {data.get('macro_over_instances', {}).get('mean_dice', 0.0):.4f}",
                    f"Mean F1: {data.get('macro_over_instances', {}).get('mean_f1', 0.0):.4f}",
                ]
            )
        )
    return "```text\n" + "\n\n".join(blocks) + "\n```"


def _benchmark_summary_html() -> str:
    parsed = _parse_output_metrics()
    if parsed is None:
        json_files = _find_metrics_json_files()
        subtitle = (
            f"Found {len(json_files)} saved metrics file(s) under inference outputs."
            if json_files
            else "Run compare_models.py or inference scripts to populate this dashboard."
        )
        return _summary_cards_html(
            "Benchmark Results",
            subtitle,
            [
                ("Status", "No parsed benchmark table yet", "warn"),
                ("Saved Images", str(len(_benchmark_gallery())), "neutral"),
                ("Inference Runs", str(len(json_files)), "neutral"),
            ],
        )

    metric_names = list(parsed["metrics"].keys())
    primary_metric = metric_names[0]
    cards = []
    for model in parsed["models"]:
        value = parsed["metrics"].get(primary_metric, {}).get(model)
        cards.append(
            (
                f"{model} {primary_metric}",
                f"{value:.4f}" if value is not None else "N/A",
                "ft" if "Fine" in model else ("base" if "ViT-S" in model else "neutral"),
            )
        )
    cards.append(("Evaluated Batches", parsed["total_batches"].split(":")[-1].strip(), "neutral"))
    return _summary_cards_html("Saved Benchmark Results", parsed["total_batches"], cards)


def _benchmark_main_chart() -> Optional[np.ndarray]:
    parsed = _parse_output_metrics()
    if parsed is None:
        return None

    categories = list(parsed["metrics"].keys())
    series = []
    palette = [(99, 102, 241), (59, 130, 246), (16, 185, 129)]
    for idx, model in enumerate(parsed["models"]):
        values = [parsed["metrics"].get(metric, {}).get(model) for metric in categories]
        series.append((model, values, palette[idx % len(palette)]))

    return _grouped_bar_chart(
        "Benchmark Leaderboard",
        categories,
        series,
        "A quick scan of saved evaluation metrics across all compared model variants.",
    )


def _benchmark_delta_chart() -> Optional[np.ndarray]:
    parsed = _parse_output_metrics()
    if parsed is None:
        return None

    if "ViT-S" not in parsed["models"] or "Fine-Tuned" not in parsed["models"]:
        return None

    categories = list(parsed["metrics"].keys())
    improvement = []
    for metric in categories:
        base = parsed["metrics"].get(metric, {}).get("ViT-S")
        ft = parsed["metrics"].get(metric, {}).get("Fine-Tuned")
        improvement.append((ft - base) if base is not None and ft is not None else None)

    return _grouped_bar_chart(
        "Fine-Tuned Gain Over ViT-S",
        categories,
        [("Gain", improvement, ACCENT_COLORS["ft"])],
        "Positive bars indicate the fine-tuned checkpoint outperforming the baseline ViT-S model.",
    )


def _benchmark_gallery() -> List[str]:
    compare_imgs = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.png")))
    if compare_imgs:
        return compare_imgs[-18:]

    vis_imgs = sorted(glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, "*", "visualizations", "*.jpg")))
    return vis_imgs[-18:]


def refresh_benchmark_assets():
    gallery = _benchmark_gallery()
    count = len(gallery)
    status = (
        f"Loaded {count} saved comparison visual(s) from `results_comparison` or `inference_outputs`."
        if count
        else "No saved comparison images were found."
    )
    return (
        _benchmark_summary_html(),
        _benchmark_main_chart(),
        _benchmark_delta_chart(),
        _metrics_markdown(),
        gallery,
        status,
    )


CUSTOM_CSS = """
.gradio-container {background:
  radial-gradient(circle at top left, rgba(14,165,233,0.12), transparent 28%),
  radial-gradient(circle at top right, rgba(16,185,129,0.14), transparent 24%),
  linear-gradient(180deg, #08111f 0%, #0f172a 42%, #111827 100%);
}
.app-shell {max-width: 1500px; margin: 0 auto; padding-bottom: 32px;}
.hero {
  background:
    radial-gradient(circle at 20% 20%, rgba(255,255,255,0.18), transparent 18%),
    linear-gradient(135deg, rgba(14,116,144,0.92) 0%, rgba(30,64,175,0.96) 45%, rgba(5,150,105,0.92) 100%);
  color: white;
  border: 1px solid rgba(255,255,255,0.18);
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.32);
  border-radius: 28px;
  padding: 28px 32px;
  margin: 8px 0 18px 0;
}
.hero h1 {margin: 0 0 10px 0; font-size: 2.35rem; letter-spacing: -0.03em;}
.hero p {margin: 0; opacity: 0.94; max-width: 900px; line-height: 1.6;}
.hero-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 22px;
}
.hero-pill {
  background: rgba(255,255,255,0.11);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.16);
  border-radius: 18px;
  padding: 12px 14px;
}
.hero-pill strong {display: block; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.78;}
.hero-pill span {display: block; margin-top: 4px; font-size: 1rem;}
.panel-note {
  background: rgba(15,23,42,0.52);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 20px;
  padding: 14px 16px;
  margin-bottom: 14px;
}
.summary-shell {
  background: linear-gradient(180deg, rgba(15,23,42,0.96) 0%, rgba(17,24,39,0.96) 100%);
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 24px;
  padding: 18px;
  box-shadow: 0 18px 50px rgba(2, 8, 23, 0.24);
}
.summary-header h3 {margin: 4px 0 6px 0; font-size: 1.35rem;}
.summary-header p {margin: 0; color: #cbd5e1; line-height: 1.5;}
.eyebrow {
  color: #7dd3fc;
  text-transform: uppercase;
  font-size: 0.78rem;
  letter-spacing: 0.08em;
  font-weight: 700;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
  margin-top: 16px;
}
.metric-card {
  border-radius: 18px;
  padding: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  min-height: 92px;
}
.metric-card-base {background: linear-gradient(180deg, rgba(30,64,175,0.3), rgba(15,23,42,0.92));}
.metric-card-ft {background: linear-gradient(180deg, rgba(5,150,105,0.32), rgba(15,23,42,0.92));}
.metric-card-neutral {background: linear-gradient(180deg, rgba(71,85,105,0.28), rgba(15,23,42,0.92));}
.metric-card-warn {background: linear-gradient(180deg, rgba(180,83,9,0.34), rgba(15,23,42,0.92));}
.metric-label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #cbd5e1;}
.metric-value {font-size: 1.15rem; font-weight: 700; margin-top: 10px; color: #f8fafc; line-height: 1.35;}
"""


with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML(
            f"""
            <div class="hero">
              <h1>Satellite Image Segmentation Workbench</h1>
              <p>
                Explore segmentation with a more polished analysis workspace. Upload an aerial image,
                prompt the model with a box or points, and inspect the baseline and fine-tuned behavior
                through overlays, structured summaries, and visual metric panels.
              </p>
              <div class="hero-strip">
                <div class="hero-pill">
                  <strong>Runtime</strong>
                  <span>{DEVICE}</span>
                </div>
                <div class="hero-pill">
                  <strong>Baseline</strong>
                  <span>{BASELINE_MODEL_LABEL}</span>
                </div>
                <div class="hero-pill">
                  <strong>Checkpoint Status</strong>
                  <span>{MODEL_STATUS}</span>
                </div>
              </div>
            </div>
            """
        )

        prompt_state = gr.State(_empty_prompt_state())
        image_state = gr.State({"image_rgb": None, "base_embedding": None, "ft_embedding": None})

        with gr.Tabs():
            with gr.Tab("Interactive Segmentation"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.HTML(
                            """
                            <div class="panel-note">
                              <strong>Prompting Workflow</strong><br>
                              Use two clicks for a bounding box, or add one or more positive points to guide the baseline model.
                              Bounding box mode is the fairest comparison for the fine-tuned checkpoint.
                            </div>
                            """
                        )
                        input_image = gr.Image(
                            label="Upload Satellite / Aerial Image",
                            type="numpy",
                            sources=["upload"],
                        )
                        prompt_mode = gr.Radio(
                            ["Bounding Box", "Point Prompt"],
                            value="Bounding Box",
                            label="Prompt Mode",
                            info="Bounding box uses two clicks. Point prompt uses one or more object clicks.",
                        )
                        preview_image = gr.Image(
                            label="Prompt Canvas",
                            type="numpy",
                            interactive=False,
                        )
                        with gr.Row():
                            clear_button = gr.Button("Clear Prompt")
                            run_button = gr.Button("Run Segmentation", variant="primary")
                        status_box = gr.Textbox(
                            label="Status",
                            value="Upload a satellite or aerial image to begin.",
                            interactive=False,
                        )
                        prompt_summary = gr.Textbox(
                            label="Prompt Summary",
                            value="No prompt prepared.",
                            interactive=False,
                        )

                    with gr.Column(scale=6):
                        gr.HTML(
                            """
                            <div class="panel-note">
                              <strong>Result View</strong><br>
                              Red-tinted panels show the baseline response, green-tinted panels show the fine-tuned response,
                              and the delta map surfaces where the masks diverge.
                            </div>
                            """
                        )
                        with gr.Row():
                            base_output = gr.Image(label="EfficientSAM ViT-S Overlay", type="numpy")
                            finetuned_output = gr.Image(label="Fine-tuned Model Overlay", type="numpy")
                        difference_output = gr.Image(
                            label="Difference View",
                            type="numpy",
                        )
                        comparison_table = gr.Dataframe(
                            headers=["Metric", "Value"],
                            datatype=["str", "str"],
                            value=[["Fine-tuned model", "Waiting for inference"]],
                            row_count=(10, "dynamic"),
                            col_count=(2, "fixed"),
                            interactive=False,
                            label="Live Comparison Summary",
                        )
                        live_summary = gr.HTML(
                            value=_summary_cards_html(
                                "Live Segmentation Result",
                                "Run a prediction to populate this executive summary and the visual metric charts.",
                                [
                                    ("Prompt", "Waiting", "neutral"),
                                    ("Base Quality", "N/A", "base"),
                                    ("Fine-Tuned Quality", "N/A", "ft"),
                                ],
                            )
                        )
                        with gr.Row():
                            live_metric_chart = gr.Image(
                                label="Interactive Metric Chart",
                                type="numpy",
                                interactive=False,
                            )
                            live_delta_chart = gr.Image(
                                label="Delta Insight Chart",
                                type="numpy",
                                interactive=False,
                            )

                input_image.change(
                    on_image_change,
                    inputs=[input_image, prompt_mode],
                    outputs=[preview_image, prompt_state, image_state, status_box],
                )
                prompt_mode.change(
                    on_mode_change,
                    inputs=[prompt_mode, input_image],
                    outputs=[preview_image, prompt_state, status_box],
                )
                preview_image.select(
                    on_preview_click,
                    inputs=[input_image, prompt_mode, prompt_state],
                    outputs=[preview_image, prompt_state, status_box],
                )
                clear_button.click(
                    clear_prompts,
                    inputs=[input_image, prompt_mode],
                    outputs=[preview_image, prompt_state, status_box],
                )
                run_button.click(
                    run_segmentation,
                    inputs=[prompt_mode, image_state, prompt_state],
                    outputs=[
                        status_box,
                        base_output,
                        finetuned_output,
                        difference_output,
                        comparison_table,
                        prompt_summary,
                        live_summary,
                        live_metric_chart,
                        live_delta_chart,
                    ],
                )

            with gr.Tab("Results Comparison"):
                gr.HTML(
                    """
                    <div class="panel-note">
                      <strong>Saved Benchmark Dashboard</strong><br>
                      This section reads your existing benchmark artifacts from <code>results_comparison</code> and
                      <code>inference_outputs</code>, then turns them into a more legible performance story.
                    </div>
                    """
                )
                benchmark_summary = gr.HTML(value=_benchmark_summary_html())
                with gr.Row():
                    benchmark_chart = gr.Image(
                        value=_benchmark_main_chart(),
                        label="Benchmark Leaderboard Chart",
                        type="numpy",
                        interactive=False,
                    )
                    benchmark_delta = gr.Image(
                        value=_benchmark_delta_chart(),
                        label="Fine-Tuned Gain Chart",
                        type="numpy",
                        interactive=False,
                    )
                with gr.Row():
                    benchmark_metrics = gr.Markdown(value=_metrics_markdown())
                    benchmark_status = gr.Textbox(
                        label="Saved Results Status",
                        value="Ready to load saved benchmark outputs.",
                        interactive=False,
                    )
                benchmark_gallery = gr.Gallery(
                    value=_benchmark_gallery(),
                    label="Benchmark Visualization Gallery",
                    columns=4,
                    object_fit="contain",
                    height=640,
                )
                refresh_button = gr.Button("Refresh Saved Results")
                refresh_button.click(
                    refresh_benchmark_assets,
                    outputs=[
                        benchmark_summary,
                        benchmark_chart,
                        benchmark_delta,
                        benchmark_metrics,
                        benchmark_gallery,
                        benchmark_status,
                    ],
                )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
