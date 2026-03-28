from __future__ import annotations

import torch
from torch import nn


class BoundaryHead(nn.Module):
    """Lightweight boundary prediction head for decoder feature maps."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        hidden_channels = max(in_channels // 2, 1)
        self.boundary_predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Shared decoder feature map with shape (B, C, H, W).

        Returns:
            Boundary probability map with shape (B, 1, H, W).
        """
        return self.boundary_predictor(features)
