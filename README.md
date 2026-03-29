# Efficient_SAM_Upgrade: Common Project README

## Overview
This repository contains two complementary EfficientSAM efforts:

1. `FIne-tuned_ISAID/`: fine-tuning EfficientSAM for iSAID aerial instance segmentation.
2. `Architectural_Changes/`: boundary-aware decoder modification of EfficientSAM for Cityscapes-style semantic segmentation.

Together, these folders capture two directions:
- dataset/domain adaptation (iSAID fine-tuning),
- architecture-level decoder enhancement (boundary-aware EfficientSAM).

## Repository Structure
```text
Efficient_SAM_Upgrade/
|-- README.md                          (this common overview)
|-- FIne-tuned_ISAID/                  (iSAID fine-tuning pipeline)
|   |-- README.md
|   |-- prepare_data.py
|   |-- dataset.py
|   |-- model_setup.py
|   |-- train.py
|   |-- compare_models.py
|   |-- checkpoints/
|   `-- results_comparison/
`-- Architectural_Changes/             (boundary-aware EfficientSAM variant)
    |-- README.md
    |-- compare.txt
    |-- compare.png
    |-- EfficientSAM/                  (proposed modified model)
    `-- EfficientSAM_original_finetune/ (baseline reference)
```

## Track 1: FIne-tuned_ISAID
### Goal
Adapt EfficientSAM to dense, small-object aerial scenes in iSAID by:
- fine-tuning the mask decoder,
- keeping image encoder and prompt encoder mostly frozen,
- optionally adding LoRA adapters to the frozen encoder.

### Workflow
```text
Raw iSAID -> patching (prepare_data.py) -> dataset_patched
-> training (train.py) -> best checkpoint
-> comparison (compare_models.py) -> quantitative + visual outputs
```

### Core implementation files
- `FIne-tuned_ISAID/prepare_data.py`
- `FIne-tuned_ISAID/dataset.py`
- `FIne-tuned_ISAID/model_setup.py`
- `FIne-tuned_ISAID/train.py`
- `FIne-tuned_ISAID/compare_models.py`
- `FIne-tuned_ISAID/Real-life use case/app.py`

### Reported benchmark (from `FIne-tuned_ISAID/results_comparison/output.txt`)
Evaluated batches: `2000` (batch size `1`)

| Metric | EfficientSAM ViT-Ti | EfficientSAM ViT-S | Fine-Tuned |
|---|---:|---:|---:|
| Mean IoU | 0.4495 | 0.4947 | 0.5361 |
| Dice (F1 Score) | 0.5380 | 0.5887 | 0.6579 |
| Precision | 0.5875 | 0.6122 | 0.6912 |
| Recall | 0.5717 | 0.6475 | 0.6986 |
| Pixel Accuracy | 0.9973 | 0.9974 | 0.9972 |

### Result summary
- Fine-tuned model improves IoU, Dice, Precision, and Recall over both base variants.
- Pixel accuracy is similar across variants, as expected for sparse aerial targets.

## Track 2: Architectural_Changes
### Goal
Improve boundary precision in EfficientSAM decoder by adding explicit edge modeling, while keeping architecture lightweight.

### Main architectural updates
- Added a dedicated boundary head:
  - `Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> Sigmoid`
- Built learnable mask-boundary fusion:
  - `Conv2d(2 -> 1, kernel_size=1)` after concatenating mask and boundary channels.
- Added Sobel-based boundary supervision from ground-truth labels.
- Used joint optimization:
  - `L_total = L_seg + lambda * L_boundary`

### Core implementation files
- `Architectural_Changes/EfficientSAM/efficient_sam/models/mask_decoder.py`
- `Architectural_Changes/EfficientSAM/efficient_sam/models/boundary_head.py`
- `Architectural_Changes/EfficientSAM/efficient_sam/utils/boundary_utils.py`
- `Architectural_Changes/EfficientSAM/efficient_sam/losses/boundary_loss.py`
- `Architectural_Changes/EfficientSAM/train.py`
- `Architectural_Changes/EfficientSAM/infer_val.py`

### Reported benchmark (from `Architectural_Changes/compare.txt`)
| Metric | Baseline (Finetuned Original) | Proposed (Novelty) | Delta (Proposed - Baseline) |
|---|---:|---:|---:|
| mIoU | 0.4251 | 0.3848 | -0.0403 |
| mDice | 0.5001 | 0.4526 | -0.0475 |
| Pixel Accuracy | 0.8271 | 0.8878 | +0.0607 |
| Boundary-IoU | 0.4641 | 0.4802 | +0.0161 |
| Boundary-F1 | 0.6297 | 0.6452 | +0.0155 |

### Result summary
- Boundary metrics improved (Boundary-IoU, Boundary-F1).
- Pixel accuracy improved.
- Region overlap metrics (mIoU, mDice) dropped in this run, indicating a boundary-region tradeoff to tune further.

## Datasets
- iSAID: used in `FIne-tuned_ISAID/`.
- Cityscapes (Kaggle mirror): https://www.kaggle.com/datasets/shuvoalok/cityscapes

## Visual Outputs
- iSAID visual comparisons:
  - `FIne-tuned_ISAID/results_comparison/compare_*.png`
- Boundary-aware architecture comparison:
  - `Architectural_Changes/compare.png`

## Detailed READMEs
For full setup and run instructions, use:
- `FIne-tuned_ISAID/README.md`
- `Architectural_Changes/README.md`

## Quick Navigation
- iSAID fine-tuning entrypoint: `FIne-tuned_ISAID/train.py`
- iSAID comparison script: `FIne-tuned_ISAID/compare_models.py`
- iSAID demo app: `FIne-tuned_ISAID/Real-life use case/app.py`
- Architectural changes training entrypoint: `Architectural_Changes/EfficientSAM/train.py`
- Architectural changes evaluation script: `Architectural_Changes/EfficientSAM/infer_val.py`
