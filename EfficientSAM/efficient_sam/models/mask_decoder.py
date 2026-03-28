# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn

from ..mlp import MLPBlock
from .boundary_head import BoundaryHead


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.invalid_points = nn.Embedding(1, embed_dim)
        self.point_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_top_left_embeddings = nn.Embedding(1, embed_dim)
        self.bbox_bottom_right_embeddings = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        invalid_label_ids = torch.eq(labels, -1)[:, :, None]
        point_label_ids = torch.eq(labels, 1)[:, :, None]
        topleft_label_ids = torch.eq(labels, 2)[:, :, None]
        bottomright_label_ids = torch.eq(labels, 3)[:, :, None]
        point_embedding = (
            point_embedding + self.invalid_points.weight[:, None, :] * invalid_label_ids
        )
        point_embedding = (
            point_embedding + self.point_embeddings.weight[:, None, :] * point_label_ids
        )
        point_embedding = (
            point_embedding
            + self.bbox_top_left_embeddings.weight[:, None, :] * topleft_label_ids
        )
        point_embedding = (
            point_embedding
            + self.bbox_bottom_right_embeddings.weight[:, None, :] * bottomright_label_ids
        )
        return point_embedding

    def forward(
        self,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning sparse prompt embeddings.

        Arguments:
          coords: A tensor of shape [B, num_points, 2]
          labels: An integer tensor of shape [B, num_points]

        Returns:
          torch.Tensor: sparse embeddings with shape BxNx(embed_dim).
        """
        return self._embed_points(coords, labels)


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones([h, w], device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int,
        activation: Type[nn.Module],
        normalization_type: str,
        normalize_before_activation: bool,
        iou_head_depth: int,
        iou_head_hidden_dim: int,
        upscaling_layer_dims: List[int],
        enable_boundary_branch: bool = False,
        semantic_num_classes: int = 1,
        enable_semantic_head: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings.

        When `enable_boundary_branch` is True, a lightweight boundary branch is
        added to the decoder and fused with mask logits via a 1x1 conv.
        """
        del normalization_type, normalize_before_activation
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.enable_boundary_branch = enable_boundary_branch
        self.semantic_num_classes = semantic_num_classes
        self.enable_semantic_head = enable_semantic_head or semantic_num_classes > 1

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        if num_multimask_outputs > 1:
            self.num_mask_tokens = num_multimask_outputs + 1
        else:
            self.num_mask_tokens = 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        output_dim_after_upscaling = transformer_dim

        self.final_output_upscaling_layers = nn.ModuleList([])
        for idx, layer_dims in enumerate(upscaling_layer_dims):
            self.final_output_upscaling_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        output_dim_after_upscaling,
                        layer_dims,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.GroupNorm(1, layer_dims)
                    if idx < len(upscaling_layer_dims) - 1
                    else nn.Identity(),
                    activation(),
                )
            )
            output_dim_after_upscaling = layer_dims

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLPBlock(
                    input_dim=transformer_dim,
                    hidden_dim=transformer_dim,
                    output_dim=output_dim_after_upscaling,
                    num_layers=2,
                    act=activation,
                )
                for _ in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_tokens,
            num_layers=iou_head_depth,
            act=activation,
        )

        if self.enable_boundary_branch:
            self.boundary_head = BoundaryHead(output_dim_after_upscaling)
            self.mask_boundary_fusion = nn.Conv2d(2, 1, kernel_size=1)
            with torch.no_grad():
                self.mask_boundary_fusion.weight.zero_()
                self.mask_boundary_fusion.bias.zero_()
                self.mask_boundary_fusion.weight[0, 0, 0, 0] = 1.0
        else:
            self.boundary_head = None
            self.mask_boundary_fusion = None

        if self.enable_semantic_head:
            self.semantic_head = nn.Conv2d(
                output_dim_after_upscaling, self.semantic_num_classes, kernel_size=1
            )
            if self.enable_boundary_branch:
                self.semantic_boundary_fusion = nn.Conv2d(2, 1, kernel_size=1)
                with torch.no_grad():
                    self.semantic_boundary_fusion.weight.zero_()
                    self.semantic_boundary_fusion.bias.zero_()
                    self.semantic_boundary_fusion.weight[0, 0, 0, 0] = 1.0
            else:
                self.semantic_boundary_fusion = None
        else:
            self.semantic_head = None
            self.semantic_boundary_fusion = None

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        return_boundary: bool = False,
        return_semantic: bool = False,
    ):
        """
        Predict masks given image and prompt embeddings.

        Returns:
          If return_boundary is False:
            (masks, iou_predictions)
          If return_boundary is True and return_semantic is False:
            (masks, iou_predictions, boundaries)
          If return_boundary and return_semantic are True:
            (masks, iou_predictions, boundaries, semantic_logits)
        """
        (
            batch_size,
            max_num_queries,
            sparse_embed_dim_1,
            sparse_embed_dim_2,
        ) = sparse_prompt_embeddings.shape

        (
            _,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        ) = image_embeddings.shape

        image_embeddings_tiled = torch.tile(
            image_embeddings[:, None, :, :, :], [1, max_num_queries, 1, 1, 1]
        ).view(
            batch_size * max_num_queries,
            image_embed_dim_c,
            image_embed_dim_h,
            image_embed_dim_w,
        )
        sparse_prompt_embeddings = sparse_prompt_embeddings.reshape(
            batch_size * max_num_queries, sparse_embed_dim_1, sparse_embed_dim_2
        )
        masks, iou_pred, boundaries, semantic_logits = self.predict_masks(
            image_embeddings=image_embeddings_tiled,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            return_boundary=return_boundary,
            return_semantic=return_semantic,
        )
        if multimask_output and self.num_multimask_outputs > 1:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
            if boundaries is not None:
                boundaries = boundaries[:, 1:, :, :]
        else:
            masks = masks[:, :1, :, :]
            iou_pred = iou_pred[:, :1]
            if boundaries is not None:
                boundaries = boundaries[:, :1, :, :]

        if return_boundary and return_semantic:
            if boundaries is None:
                boundaries = torch.zeros_like(masks)
            if semantic_logits is None:
                semantic_logits = torch.zeros(
                    masks.shape[0],
                    self.semantic_num_classes,
                    masks.shape[-2],
                    masks.shape[-1],
                    device=masks.device,
                    dtype=masks.dtype,
                )
            return masks, iou_pred, boundaries, semantic_logits
        if return_boundary:
            if boundaries is None:
                boundaries = torch.zeros_like(masks)
            return masks, iou_pred, boundaries
        if return_semantic:
            if semantic_logits is None:
                semantic_logits = torch.zeros(
                    masks.shape[0],
                    self.semantic_num_classes,
                    masks.shape[-2],
                    masks.shape[-1],
                    device=masks.device,
                    dtype=masks.dtype,
                )
            return masks, iou_pred, semantic_logits
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        return_boundary: bool = False,
        return_semantic: bool = False,
    ):
        """Predicts masks. See `forward` for details."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape
        hs, src = self.transformer(image_embeddings, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        upscaled_embedding = src.transpose(1, 2).view(b, c, h, w)
        for upscaling_layer in self.final_output_upscaling_layers:
            upscaled_embedding = upscaling_layer(upscaled_embedding)

        hyper_in_list: List[torch.Tensor] = []
        for idx, output_hypernetworks_mlp in enumerate(self.output_hypernetworks_mlps):
            hyper_in_list.append(output_hypernetworks_mlp(mask_tokens_out[:, idx, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        boundaries: Optional[torch.Tensor] = None
        boundary_single: Optional[torch.Tensor] = None
        if self.enable_boundary_branch and self.boundary_head is not None:
            boundary_single = self.boundary_head(upscaled_embedding)
            boundary_map = boundary_single.expand(-1, masks.shape[1], -1, -1)
            if self.mask_boundary_fusion is not None:
                fusion_input = torch.stack((masks, boundary_map), dim=2).reshape(
                    b * masks.shape[1], 2, h, w
                )
                masks = self.mask_boundary_fusion(fusion_input).view(
                    b, masks.shape[1], h, w
                )
            boundaries = boundary_map

        semantic_logits: Optional[torch.Tensor] = None
        if self.enable_semantic_head and self.semantic_head is not None:
            semantic_logits = self.semantic_head(upscaled_embedding)
            if (
                boundary_single is not None
                and self.semantic_boundary_fusion is not None
            ):
                semantic_boundary = boundary_single.expand(
                    -1, self.semantic_num_classes, -1, -1
                )
                semantic_fusion_input = torch.stack(
                    (semantic_logits, semantic_boundary), dim=2
                ).reshape(b * self.semantic_num_classes, 2, h, w)
                semantic_logits = self.semantic_boundary_fusion(
                    semantic_fusion_input
                ).view(b, self.semantic_num_classes, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if return_boundary or return_semantic:
            return masks, iou_pred, boundaries, semantic_logits
        return masks, iou_pred, None, None
