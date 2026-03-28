# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PIXEL_MEAN, PIXEL_STD, iSAIDPatchDataset
from model_setup import freeze_model, inject_lora, load_efficient_sam


def compute_all_metrics(pred_logits, gt_masks):
    pred = (pred_logits > 0.0).float()
    gt = gt_masks.float()

    tp = (pred * gt).sum(dim=(1, 2, 3))
    fp = (pred * (1 - gt)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * gt).sum(dim=(1, 2, 3))
    tn = ((1 - pred) * (1 - gt)).sum(dim=(1, 2, 3))

    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    acc = (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "acc": acc.mean().item(),
    }


def unnormalize_image(img_tensor):
    device = img_tensor.device
    mean = torch.tensor(PIXEL_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(PIXEL_STD, device=device).view(3, 1, 1)
    img = img_tensor * std + mean
    img = img.clamp(0, 1).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255).astype(np.uint8)


def denormalize_batch_for_model(images):
    mean = torch.tensor(PIXEL_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(PIXEL_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


def overlay_mask(image, mask_tensor, color, alpha=0.5):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    for channel in range(3):
        overlay[:, :, channel] = np.where(mask_resized == 1, color[channel], overlay[:, :, channel])

    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def load_finetuned_model(ckpt_path, device):
    print("Loading Fine-Tuned Model from {}...".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=device)
    model_type = ckpt.get("model_type", "vits")
    model = load_efficient_sam(model_type, device=device)
    freeze_model(model)
    model.mask_decoder.load_state_dict(ckpt["mask_decoder"])

    if ckpt.get("use_lora", False):
        inject_lora(model, rank=4, alpha=1.0)
        for name, param in model.image_encoder.named_parameters():
            if name in ckpt.get("lora_params", {}):
                param.data.copy_(ckpt["lora_params"][name])

    model = model.to(device)
    model.eval()
    return model, model_type


def build_panel(image_rgb, mask_tensor, title, text_color, mask_color):
    panel = overlay_mask(image_rgb, mask_tensor, color=mask_color, alpha=0.5)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.putText(
        panel,
        title,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        3,
    )
    return panel


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Compare GT with EfficientSAM ViT-Ti, ViT-S, and the fine-tuned checkpoint."
    )
    parser.add_argument("--data_root", type=str, default="dataset_patched")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test' or 'val')")
    parser.add_argument("--ckpt", type=str, default="checkpoints/best_mask_decoder.pth")
    parser.add_argument("--out_dir", type=str, default="results_comparison")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images/batches to compare (0 for ALL)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--no_vis", action="store_true", help="Disable saving PNG comparison grids")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_device_type = "cuda" if device.startswith("cuda") else "cpu"
    print("Device: {}".format(device))
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading EfficientSAM ViT-Ti baseline...")
    vit_ti_model = load_efficient_sam("vitt", device=device)
    vit_ti_model.eval()

    print("Loading EfficientSAM ViT-S baseline...")
    vit_s_model = load_efficient_sam("vits", device=device)
    vit_s_model.eval()

    ft_model, ft_model_type = load_finetuned_model(args.ckpt, device)

    print("Loading split '{}' from {} ...".format(args.split, args.data_root))
    test_ds = iSAIDPatchDataset(args.data_root, split=args.split)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_specs = [
        {
            "key": "vit_ti",
            "label": "EfficientSAM ViT-Ti",
            "model": vit_ti_model,
            "input_mode": "native",
            "use_multimask": True,
        },
        {
            "key": "vit_s",
            "label": "EfficientSAM ViT-S",
            "model": vit_s_model,
            "input_mode": "native",
            "use_multimask": True,
        },
        {
            "key": "finetuned",
            "label": "Fine-Tuned ({})".format(ft_model_type),
            "model": ft_model,
            "input_mode": "legacy_finetuned",
            "use_multimask": False,
        },
    ]
    metric_names = ["iou", "dice", "precision", "recall", "acc"]
    metric_sums = {spec["key"]: {name: 0.0 for name in metric_names} for spec in model_specs}

    count = 0
    total_len = args.max_images if args.max_images > 0 else len(test_loader)
    print("\nGenerating evaluations for {} batches...".format(total_len))
    pbar = tqdm(total=total_len)

    for images, gt_masks, bbox_points, bbox_labels in test_loader:
        if args.max_images > 0 and count >= args.max_images:
            break

        images = images.to(device)
        gt_masks = gt_masks.to(device)
        bbox_points = bbox_points.to(device)
        bbox_labels = bbox_labels.to(device)
        native_model_images = denormalize_batch_for_model(images)

        predictions = {}
        with autocast(device_type=amp_device_type):
            for spec in model_specs:
                model_images = native_model_images if spec["input_mode"] == "native" else images
                image_embeddings = spec["model"].get_image_embeddings(model_images)
                pred_masks, iou_scores = spec["model"].predict_masks(
                    image_embeddings=image_embeddings,
                    batched_points=bbox_points,
                    batched_point_labels=bbox_labels,
                    multimask_output=spec["use_multimask"],
                    input_h=1024,
                    input_w=1024,
                )
                if spec["use_multimask"]:
                    candidate_scores = iou_scores.squeeze(1)
                    best_idx = torch.argmax(candidate_scores, dim=1)
                    batch_indices = torch.arange(pred_masks.shape[0], device=pred_masks.device)
                    pred_masks = pred_masks[batch_indices, 0, best_idx].unsqueeze(1)
                else:
                    pred_masks = pred_masks[:, 0, 0].unsqueeze(1)
                pred_binary = (pred_masks > 0.0).float()
                metrics = compute_all_metrics(pred_masks, gt_masks)

                predictions[spec["key"]] = {
                    "mask_logits": pred_masks,
                    "binary": pred_binary,
                    "metrics": metrics,
                }
                for metric_name in metric_names:
                    metric_sums[spec["key"]][metric_name] += metrics[metric_name]

        if not args.no_vis:
            img_rgb = unnormalize_image(images[0])
            gt_mask = gt_masks[0]

            panels = [
                build_panel(
                    img_rgb,
                    gt_mask,
                    "Ground Truth (GT)",
                    text_color=(0, 255, 0),
                    mask_color=(0, 255, 0),
                )
            ]

            vis_styles = {
                "vit_ti": {"text_color": (0, 165, 255), "mask_color": (255, 140, 0)},
                "vit_s": {"text_color": (255, 0, 0), "mask_color": (255, 0, 0)},
                "finetuned": {"text_color": (0, 255, 255), "mask_color": (0, 255, 255)},
            }

            for spec in model_specs:
                result = predictions[spec["key"]]
                title = "{} (IoU: {:.2f})".format(spec["label"], result["metrics"]["iou"])
                panels.append(
                    build_panel(
                        img_rgb,
                        result["binary"][0],
                        title,
                        text_color=vis_styles[spec["key"]]["text_color"],
                        mask_color=vis_styles[spec["key"]]["mask_color"],
                    )
                )

            combined = np.hstack(panels)
            save_path = os.path.join(args.out_dir, "compare_{:04d}.png".format(count))
            cv2.imwrite(save_path, combined)

        count += 1
        pbar.update(1)

    pbar.close()

    if count == 0:
        raise RuntimeError("No batches were evaluated. Check your split/data_root arguments.")

    metrics_display = {
        "iou": "Mean IoU",
        "dice": "Dice (F1 Score)",
        "precision": "Precision",
        "recall": "Recall",
        "acc": "Pixel Accuracy",
    }

    report = []
    report.append("=================================================================")
    report.append("               GT VS VIT-TI VS VIT-S VS FINE-TUNED")
    report.append("=================================================================")
    report.append("")
    report.append("Total batches evaluated: {} (Batch Size: {})".format(count, args.batch_size))
    report.append("")
    report.append(
        "{:<20} | {:<15} | {:<15} | {:<15}".format(
            "Metric", "ViT-Ti", "ViT-S", "Fine-Tuned"
        )
    )
    report.append("-" * 75)

    for metric_name, display_name in metrics_display.items():
        vit_ti_val = metric_sums["vit_ti"][metric_name] / count
        vit_s_val = metric_sums["vit_s"][metric_name] / count
        finetuned_val = metric_sums["finetuned"][metric_name] / count
        report.append(
            "{:<20} | {:<15.4f} | {:<15.4f} | {:<15.4f}".format(
                display_name,
                vit_ti_val,
                vit_s_val,
                finetuned_val,
            )
        )

    report.append("-" * 75)
    report.append("")
    report.append("Ground Truth (GT) is shown in every saved comparison image as the first panel.")
    report_text = "\n".join(report)

    print("\n" + report_text)

    out_txt_path = os.path.join(args.out_dir, "output.txt")
    with open(out_txt_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)
    print("Results successfully saved to: {}\n".format(out_txt_path))


if __name__ == "__main__":
    main()
