# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .efficient_sam import build_efficient_sam

def build_efficient_sam_vitt(
    enable_boundary_decoder: bool = False,
    semantic_num_classes: int = 1,
    enable_semantic_head: bool = False,
):
    return build_efficient_sam(
        encoder_patch_embed_dim=192,
        encoder_num_heads=3,
        checkpoint="weights/efficient_sam_vitt.pt",
        enable_boundary_decoder=enable_boundary_decoder,
        semantic_num_classes=semantic_num_classes,
        enable_semantic_head=enable_semantic_head,
    ).eval()


def build_efficient_sam_vits(
    enable_boundary_decoder: bool = False,
    semantic_num_classes: int = 1,
    enable_semantic_head: bool = False,
):
    return build_efficient_sam(
        encoder_patch_embed_dim=384,
        encoder_num_heads=6,
        checkpoint="weights/efficient_sam_vits.pt",
        enable_boundary_decoder=enable_boundary_decoder,
        semantic_num_classes=semantic_num_classes,
        enable_semantic_head=enable_semantic_head,
    ).eval()
