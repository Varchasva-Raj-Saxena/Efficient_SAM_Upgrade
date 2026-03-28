# EfficientSAM Fine-tuning on iSAID

Fine-tune [EfficientSAM](https://github.com/yformer/EfficientSAM) on the **iSAID** (Instance Segmentation in Aerial Images) dataset to adapt the model to top-down aerial imagery with tiny objects.

## Pipeline Overview

```
prepare_data.py  →  dataset.py  →  model_setup.py  →  train.py
   (slice)          (load)          (init/freeze)      (train)
```

| File | Purpose |
|------|---------|
| `prepare_data.py` | Slice large aerial images into 1024×1024 patches; recalculate annotations; filter background |
| `dataset.py` | PyTorch Dataset with instance mask loading, noisy bbox prompt generation |
| `model_setup.py` | Load EfficientSAM, freeze encoder/prompt-encoder, optional LoRA injection |
| `train.py` | Training loop: Dice + Focal loss, AdamW, cosine LR, validation IoU, checkpointing |

---

## 1. Environment Setup

```bash
pip install torch torchvision torchaudio
pip install opencv-python pycocotools tqdm numpy
# Optional (for LoRA and advanced augmentation):
pip install albumentations sahi
```

## 2. Dataset Preparation

Make sure your dataset folder looks like this:

```
dataset/
├── train/
│   ├── Annotations/iSAID_train.json
│   ├── Images/          ← raw aerial PNGs
│   └── Instance_masks/  ← P*_instance_id_RGB.png
├── val/
│   ├── Annotations/iSAID_val.json
│   ├── Images/
│   └── Instance_masks/
└── test/
    ├── Annotations/
    └── Images/
```

## 3. Slice Images into Patches

```bash
python prepare_data.py \
    --data_root dataset \
    --output_root dataset_patched \
    --patch_size 1024 \
    --stride 512 \
    --bg_keep_ratio 0.1
```

This creates:

```
dataset_patched/
├── train/
│   ├── Images/           ← 1024×1024 patches
│   ├── Instance_masks/   ← corresponding mask patches
│   └── annotations.json  ← COCO-format annotations
└── val/
    ├── Images/
    ├── Instance_masks/
    └── annotations.json
```

## 4. Train the Model

### Basic (Mask Decoder only)

```bash
python train.py \
    --data_root dataset_patched \
    --epochs 25 \
    --batch_size 4 \
    --lr 1e-4 \
    --model_type vits
```

### With LoRA (recommended for better domain adaptation)

```bash
python train.py \
    --data_root dataset_patched \
    --epochs 25 \
    --batch_size 4 \
    --lr 1e-4 \
    --model_type vits \
    --use_lora \
    --lora_rank 4 \
    --lora_alpha 1.0
```

### Resume from a checkpoint for more epochs

```bash
python train.py \
    --data_root dataset_patched \
    --epochs 25 \
    --batch_size 16 \
    --lr 1e-4 \
    --model_type vits \
    --use_lora \
    --lora_rank 4 \
    --resume_checkpoint checkpoints/latest_checkpoint.pth \
    --extra_epochs 10
```

### All CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `dataset_patched` | Path to patched dataset |
| `--epochs` | `25` | Number of training epochs |
| `--batch_size` | `4` | Batch size |
| `--lr` | `1e-4` | Learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--model_type` | `vits` | `vits` (ViT-S) or `vitt` (ViT-Ti) |
| `--use_lora` | `False` | Enable LoRA in image encoder |
| `--lora_rank` | `4` | LoRA rank |
| `--lora_alpha` | `1.0` | LoRA scaling factor |
| `--save_dir` | `checkpoints` | Directory for saved models |
| `--device` | auto | `cuda` or `cpu` |
| `--resume_checkpoint` | `None` | Resume training from a saved checkpoint |
| `--extra_epochs` | `0` | When resuming, train this many more epochs beyond the checkpoint epoch |

## 5. Load Best Checkpoint for Inference

```python
import torch
from model_setup import load_efficient_sam, freeze_model, inject_lora

# Rebuild the model
model = load_efficient_sam("vits", device="cuda")
freeze_model(model)

# Load best weights
ckpt = torch.load("checkpoints/best_mask_decoder.pth", map_location="cuda")
model.mask_decoder.load_state_dict(ckpt["mask_decoder"])

# If LoRA was used during training
if ckpt.get("use_lora"):
    inject_lora(model, rank=4, alpha=1.0)
    for name, param in model.image_encoder.named_parameters():
        if name in ckpt["lora_params"]:
            param.data = ckpt["lora_params"][name]

model.eval()
```

## 6. Create a Small Patched Test Set (200 Images)

The official `dataset/test` split has no ground-truth instance annotations, so
metrics like IoU/Dice cannot be computed there. For quantitative evaluation,
create a small labeled test split from `dataset_patched/val`:

```bash
python create_test_dataset_patched.py \
    --source_root dataset_patched/val \
    --output_root test_dataset_patched \
    --num_images 200 \
    --seed 42
```

This creates:

```
test_dataset_patched/
└── test/
    ├── Images/
    ├── Instance_masks/
    └── annotations.json
```

## 7. Inference: Base Model vs Fine-tuned Model

### Base pretrained model

```bash
python inference_base.py \
    --dataset_root test_dataset_patched \
    --split test \
    --model_type vits \
    --output_root inference_outputs
```

### Fine-tuned model

```bash
python inference_finetuned.py \
    --dataset_root test_dataset_patched \
    --split test \
    --model_type vits \
    --checkpoint checkpoints/best_mask_decoder.pth \
    --lora_rank 4 \
    --lora_alpha 1.0 \
    --output_root inference_outputs
```

Each run writes:

```
inference_outputs/<run_name>/
├── metrics_summary.json
├── per_instance_metrics.csv
├── pred_masks/
└── visualizations/
```

The visualization panels include:
1. Original image
2. GT mask overlay (green)
3. Predicted mask overlay (red)
4. Error map (TP/FP/FN)

## Loss Design

- **Focal Loss** (×20): handles extreme foreground/background imbalance in aerial imagery
- **Dice Loss** (×1): encourages sharp mask boundaries

## Architecture

```
Image ──► [Frozen ViT Encoder (+optional LoRA)] ──► Image Embeddings ──┐
                                                                       ├──► [Trainable Mask Decoder] ──► Predicted Mask
Noisy BBox ──► [Frozen Prompt Encoder] ──► Sparse Embeddings ──────────┘
```
