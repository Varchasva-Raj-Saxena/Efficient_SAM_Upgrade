"""
model_setup.py — EfficientSAM model initialisation, weight freezing and LoRA.

Provides utilities to:
  1. Load EfficientSAM-S or EfficientSAM-Ti with pretrained weights.
  2. Freeze the Image Encoder and Prompt Encoder, keeping only the Mask
     Decoder trainable.
  3. Optionally inject LoRA adapters into the frozen Image Encoder attention
     layers for improved domain adaptation.

Usage:
    from model_setup import load_efficient_sam, freeze_model, inject_lora

    model = load_efficient_sam("vits")
    freeze_model(model)
    inject_lora(model, rank=4)  # optional
"""

import sys
import os
import math
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure the cloned EfficientSAM repo is importable
# ---------------------------------------------------------------------------
_REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "EfficientSAM")
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from efficient_sam.build_efficient_sam import (  # noqa: E402
    build_efficient_sam_vits,
    build_efficient_sam_vitt,
)


def _checkpoint_for_model_type(model_type: str) -> str:
    return os.path.join(_REPO_DIR, "weights", f"efficient_sam_{model_type}.pt")


def _assert_checkpoint_looks_valid(model_type: str) -> None:
    """
    Guard against truncated/corrupted checkpoint files.
    """
    ckpt = _checkpoint_for_model_type(model_type)
    min_expected_bytes = 5 * 1024 * 1024  # far below real size, catches bad files

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Missing EfficientSAM checkpoint: {ckpt}. "
            "Please download the model weights into EfficientSAM/weights."
        )

    size = os.path.getsize(ckpt)
    if size < min_expected_bytes:
        raise RuntimeError(
            f"Checkpoint looks incomplete: {ckpt} ({size} bytes). "
            "Please re-download EfficientSAM weights; truncated files cause "
            "PytorchStreamReader/zip archive errors."
        )


# ===================================================================
# 1. Model Loading
# ===================================================================

def load_efficient_sam(model_type: str = "vits",
                       device: str = "cpu") -> nn.Module:
    """
    Load a pretrained EfficientSAM model.

    Args:
        model_type: ``"vits"`` (ViT-S, 384-dim) or ``"vitt"`` (ViT-Ti, 192-dim)
        device: target device

    Returns:
        The loaded ``EfficientSam`` model in eval mode.
    """
    _assert_checkpoint_looks_valid(model_type)

    # We need to set cwd so relative checkpoint paths resolve correctly.
    orig_cwd = os.getcwd()
    os.chdir(_REPO_DIR)
    try:
        if model_type == "vits":
            model = build_efficient_sam_vits()
        elif model_type == "vitt":
            model = build_efficient_sam_vitt()
        else:
            raise ValueError(f"Unknown model_type: {model_type}. "
                             f"Choose 'vits' or 'vitt'.")
    finally:
        os.chdir(orig_cwd)

    model = model.to(device)
    print(f"✅ Loaded EfficientSAM-{model_type.upper()} on {device}")
    return model


# ===================================================================
# 2. Weight Freezing
# ===================================================================

def freeze_model(model: nn.Module) -> None:
    """
    Freeze the Image Encoder and Prompt Encoder.
    Only the Mask Decoder remains trainable.
    """
    # Freeze Image Encoder
    for param in model.image_encoder.parameters():
        param.requires_grad = False

    # Freeze Prompt Encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = False

    # Ensure Mask Decoder is trainable
    for param in model.mask_decoder.parameters():
        param.requires_grad = True

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔒 Frozen model — Trainable: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")


# ===================================================================
# 3. LoRA (Low-Rank Adaptation)
# ===================================================================

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for ``nn.Linear``.

    Keeps the original weight frozen and adds a low-rank trainable update:
        y = W_frozen @ x + (alpha / r) * B @ A @ x

    Args:
        original_linear: the linear layer to adapt.
        rank: LoRA rank (``r``).
        alpha: LoRA scaling factor.
    """

    def __init__(self, original_linear: nn.Linear,
                 rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Freeze original weights
        for p in self.original_linear.parameters():
            p.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Kaiming initialisation for A, zeros for B (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + self.scaling * lora_out


def inject_lora(model: nn.Module, rank: int = 4,
                alpha: float = 1.0) -> None:
    """
    Inject LoRA adapters into the ``qkv`` and ``proj`` layers of every
    attention block in the frozen Image Encoder.

    Args:
        model: an ``EfficientSam`` model (already frozen via ``freeze_model``).
        rank: LoRA rank.
        alpha: LoRA scaling factor.
    """
    lora_count = 0
    for block in model.image_encoder.blocks:
        # Replace qkv linear
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRALinear(old_qkv, rank=rank, alpha=alpha)
        lora_count += 1

        # Replace proj linear
        old_proj = block.attn.proj
        block.attn.proj = LoRALinear(old_proj, rank=rank, alpha=alpha)
        lora_count += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Injected LoRA (rank={rank}, alpha={alpha}) into "
          f"{lora_count} layers")
    print(f"   New trainable params: {trainable:,}")


# ===================================================================
# 4. Utility
# ===================================================================

def get_trainable_params(model: nn.Module):
    """Return a list of all parameters with ``requires_grad = True``."""
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = load_efficient_sam("vits", device="cpu")
    freeze_model(model)
    inject_lora(model, rank=4)
    params = get_trainable_params(model)
    print(f"\nTotal trainable parameters: {sum(p.numel() for p in params):,}")
