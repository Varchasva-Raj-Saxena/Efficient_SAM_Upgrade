# EfficientSAM Fine-Tuning for iSAID Aerial Instance Segmentation

This repository adapts [EfficientSAM](https://github.com/yformer/EfficientSAM) to the **iSAID** aerial-image dataset by fine-tuning the mask decoder and optionally adding **LoRA** adapters to the frozen image encoder. The project also includes:

- patch-based iSAID preprocessing for very large aerial images
- training and checkpointing code
- baseline vs fine-tuned quantitative comparison
- saved visual comparisons
- a Gradio demo for interactive segmentation on real images

The main project code lives in [`FIne-tuned_ISAID`](./FIne-tuned_ISAID).

## Project Overview

The training flow in this repo is:

```text
Raw iSAID dataset
    -> prepare_data.py
    -> patched 1024x1024 dataset
    -> dataset.py
    -> model_setup.py
    -> train.py
    -> checkpoints/best_mask_decoder.pth
    -> compare_models.py / Gradio demo
```

### What is being fine-tuned?

- `image_encoder`: frozen by default
- `prompt_encoder`: frozen
- `mask_decoder`: trainable
- `LoRA on image_encoder`: optional

This is a lightweight adaptation strategy that keeps EfficientSAM fast while making it much better suited to dense small-object aerial scenes.

## Repository Structure

```text
Efficient_SAM_Upgrade/
├── README.md
└── FIne-tuned_ISAID/
    ├── compare_models.py
    ├── dataset.py
    ├── model_setup.py
    ├── prepare_data.py
    ├── train.py
    ├── checkpoints/
    │   ├── best_mask_decoder.pth
    │   └── latest_checkpoint.pth
    ├── results_comparison/
    │   ├── output.txt
    │   └── compare_*.png
    └── Real-life use case/
        └── app.py
```

### Core files

- `prepare_data.py`: slices huge iSAID images into overlapping `1024x1024` patches and rewrites annotations
- `dataset.py`: loads patched images, extracts binary masks from iSAID instance masks, and creates noisy box prompts
- `model_setup.py`: loads EfficientSAM, freezes modules, and injects LoRA when enabled
- `train.py`: training loop with Dice + Focal loss, validation IoU, checkpoint saving, and resume support
- `compare_models.py`: compares pretrained EfficientSAM ViT-Ti, pretrained EfficientSAM ViT-S, and the fine-tuned checkpoint
- `Real-life use case/app.py`: Gradio app for interactive comparison on uploaded images

## 1. Clone This Repository

```bash
git clone <your-repo-url>
cd Efficient_SAM_Upgrade
```

## 2. Create the Python Environment

Use Python `3.8+` and install the required libraries:

```bash
pip install torch torchvision torchaudio
pip install numpy opencv-python pycocotools tqdm gradio
```

If you want to keep the environment isolated:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy opencv-python pycocotools tqdm gradio
```

## 3. Clone EfficientSAM Inside the Project

`model_setup.py` expects the original EfficientSAM repo to exist at:

```text
FIne-tuned_ISAID/EfficientSAM
```

Clone it there:

```bash
cd FIne-tuned_ISAID
git clone https://github.com/yformer/EfficientSAM.git
cd ..
```

After cloning, the layout should look like:

```text
FIne-tuned_ISAID/
├── EfficientSAM/
│   ├── efficient_sam/
│   └── weights/
├── train.py
├── prepare_data.py
└── ...
```

## 4. Download EfficientSAM Weights

This repo expects the pretrained checkpoints to be placed here:

```text
FIne-tuned_ISAID/EfficientSAM/weights/efficient_sam_vits.pt
FIne-tuned_ISAID/EfficientSAM/weights/efficient_sam_vitt.pt
```

`model_setup.py` will fail if these files are missing or corrupted.

If you only plan to fine-tune and evaluate the ViT-S version, the most important file is:

```text
efficient_sam_vits.pt
```

If you also want the comparison against ViT-Ti, download:

```text
efficient_sam_vitt.pt
```

## 5. Download the iSAID Dataset

Download the **iSAID** dataset from its official source and unpack it so that the raw dataset looks like this:

```text
FIne-tuned_ISAID/
└── dataset/
    ├── train/
    │   ├── Annotations/
    │   │   └── iSAID_train.json
    │   ├── Images/
    │   └── Instance_masks/
    ├── val/
    │   ├── Annotations/
    │   │   └── iSAID_val.json
    │   ├── Images/
    │   └── Instance_masks/
    └── test/
        ├── Annotations/
        ├── Images/
        └── Instance_masks/
```

Notes:

- `prepare_data.py` currently processes `train` and `val`
- the code expects the iSAID RGB instance-mask files to be present
- if your local folder names differ, rename them to match the structure above

## 6. Prepare the Patched Dataset

From inside [`FIne-tuned_ISAID`](./FIne-tuned_ISAID), run:

```bash
cd FIne-tuned_ISAID
python prepare_data.py --data_root dataset --output_root dataset_patched --patch_size 1024 --stride 512 --bg_keep_ratio 0.1
```

This creates:

```text
FIne-tuned_ISAID/
└── dataset_patched/
    ├── train/
    │   ├── Images/
    │   ├── Instance_masks/
    │   └── annotations.json
    └── val/
        ├── Images/
        ├── Instance_masks/
        └── annotations.json
```

What this step does:

- slices large aerial images into overlapping `1024x1024` patches
- carries over only the annotations that overlap each patch
- filters out most background-only patches using `bg_keep_ratio`
- saves new COCO-style `annotations.json` files for training

## 7. Train the Fine-Tuned Model

### Standard fine-tuning

```bash
python train.py --data_root dataset_patched --epochs 25 --batch_size 16 --lr 1e-4 --model_type vits
```

### Fine-tuning with LoRA

```bash
python train.py --data_root dataset_patched --epochs 25 --batch_size 16 --lr 1e-4 --model_type vits --use_lora --lora_rank 4 --lora_alpha 1.0
```

### Resume training

```bash
python train.py --data_root dataset_patched --model_type vits --use_lora --resume_checkpoint checkpoints/latest_checkpoint.pth --extra_epochs 10
```

Saved checkpoints:

- `checkpoints/best_mask_decoder.pth`
- `checkpoints/latest_checkpoint.pth`

Important training details:

- loss: `20 x focal loss + 1 x dice loss`
- validation metric: mean IoU
- default backbone choice: `vits`
- `train.py` can auto-run data preparation if `dataset_patched` annotations are missing and the raw `dataset` folder exists

## 8. Compare Base vs Fine-Tuned Models

The repo includes [`compare_models.py`](./FIne-tuned_ISAID/compare_models.py), which evaluates:

- EfficientSAM ViT-Ti
- EfficientSAM ViT-S
- the fine-tuned checkpoint

Example:

```bash
python compare_models.py --data_root dataset_patched --split val --ckpt checkpoints/best_mask_decoder.pth --out_dir results_comparison --max_images 200
```

Notes:

- in the current repo snapshot, `prepare_data.py` generates `train` and `val`, so `--split val` is the safest runnable option
- if you create your own separate labeled test split, you can point `--split test` to it instead

This produces:

```text
results_comparison/
├── compare_0000.png ...
└── output.txt
```

## 9. Launch the Interactive Demo

From inside [`FIne-tuned_ISAID`](./FIne-tuned_ISAID):

```bash
python "Real-life use case/app.py"
```

The Gradio app:

- loads the baseline EfficientSAM ViT-S model
- loads the fine-tuned checkpoint if available
- supports bounding-box and point prompts
- shows side-by-side overlays and difference maps
- reads `results_comparison/output.txt` and saved benchmark images for a dashboard view

## Benchmark Results

The following numbers are taken directly from [`FIne-tuned_ISAID/results_comparison/output.txt`](./FIne-tuned_ISAID/results_comparison/output.txt).

### Quantitative summary

Evaluated batches: `2000` with batch size `1`

| Metric | EfficientSAM ViT-Ti | EfficientSAM ViT-S | Fine-Tuned |
|---|---:|---:|---:|
| Mean IoU | 0.4495 | 0.4947 | 0.5361 |
| Dice (F1 Score) | 0.5380 | 0.5887 | 0.6579 |
| Precision | 0.5875 | 0.6122 | 0.6912 |
| Recall | 0.5717 | 0.6475 | 0.6986 |
| Pixel Accuracy | 0.9973 | 0.9974 | 0.9972 |

### What these results show

- the fine-tuned model beats both pretrained baselines on **IoU**
- it also improves **Dice**, **precision**, and **recall**
- the biggest gain is in object-level mask quality rather than pixel accuracy, which is expected for sparse aerial targets

## Visual Comparison Results

These examples are already saved in [`results_comparison`](./FIne-tuned_ISAID/results_comparison).

### Example 1

![Comparison example 1](./results_comparison/compare_0011.png)

### Example 2

![Comparison example 2](./results_comparison/compare_0059.png)

### Example 3

![Comparison example 3](./results_comparison/compare_1841.png)

Panel order in each visualization:

1. Ground truth
2. EfficientSAM ViT-Ti
3. EfficientSAM ViT-S
4. Fine-tuned model

## Output.txt Results

For convenience, the saved benchmark text is reproduced below:

```text
=================================================================
               GT VS VIT-TI VS VIT-S VS FINE-TUNED
=================================================================

Total batches evaluated: 2000 (Batch Size: 1)

Metric               | ViT-Ti          | ViT-S           | Fine-Tuned
---------------------------------------------------------------------------
Mean IoU             | 0.4495          | 0.4947          | 0.5361
Dice (F1 Score)      | 0.5380          | 0.5887          | 0.6579
Precision            | 0.5875          | 0.6122          | 0.6912
Recall               | 0.5717          | 0.6475          | 0.6986
Pixel Accuracy       | 0.9973          | 0.9974          | 0.9972
---------------------------------------------------------------------------

Ground Truth (GT) is shown in every saved comparison image as the first panel.
```

## End-to-End Quick Start

If you want the shortest path to reproducing the workflow:

```bash
git clone <your-repo-url>
cd Efficient_SAM_Upgrade
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install numpy opencv-python pycocotools tqdm gradio
cd FIne-tuned_ISAID
git clone https://github.com/yformer/EfficientSAM.git
```

Then:

1. Put EfficientSAM weights in `FIne-tuned_ISAID/EfficientSAM/weights/`
2. Put the raw iSAID dataset in `FIne-tuned_ISAID/dataset/`
3. Run `python prepare_data.py --data_root dataset --output_root dataset_patched`
4. Run `python train.py --data_root dataset_patched --model_type vits --use_lora`
5. Run `python compare_models.py --data_root dataset_patched --split val --ckpt checkpoints/best_mask_decoder.pth`
6. Run `python "Real-life use case/app.py"`

## Current Included Artifacts

This repo snapshot already includes:

- saved checkpoints under `FIne-tuned_ISAID/checkpoints/`
- saved comparison figures under `FIne-tuned_ISAID/results_comparison/`
- a saved benchmark summary in `FIne-tuned_ISAID/results_comparison/output.txt`

That means you can inspect the reported results and run the Gradio app even before retraining, as long as EfficientSAM and its pretrained weights are placed in the expected location.
