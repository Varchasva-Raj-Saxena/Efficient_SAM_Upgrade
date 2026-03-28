# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Backward-compatible decoder import shim.
from .models.mask_decoder import MaskDecoder, PositionEmbeddingRandom, PromptEncoder

__all__ = ["MaskDecoder", "PositionEmbeddingRandom", "PromptEncoder"]
