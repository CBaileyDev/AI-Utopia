"""Section 4.3 Per-role obs encoder. M1 ships gatherer only."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class GathererRoleEncoder(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.grid_conv = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
        )
        flat_in = 8 * 6 + 1 + 4 * 4   # 65
        self.flat_mlp = nn.Sequential(
            nn.Linear(flat_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        # T21 fix #9: Ray 2.55's ConnectorV2 flattens Dict obs values to 2D
        # (B, prod(shape)) before passing to the RLModule. Reshape grid back
        # to (B, 32, 32, 6) when that happens. Unit tests construct 4D obs
        # directly so they don't hit this — only real EnvRunner rollouts do.
        grid = obs["g_resource_grid"]
        if grid.ndim == 2:
            grid = grid.reshape(-1, 32, 32, 6)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        grid_feat = self.grid_conv(grid)

        nearest = obs["g_nearest_resources"]
        nearest = nearest.reshape(nearest.shape[0], -1)

        rich = obs["g_richness_score"]
        if rich.ndim == 1:
            rich = rich.unsqueeze(-1)

        host = obs["g_hostiles_nearby"]
        host = host.reshape(host.shape[0], -1)

        flat    = torch.cat([nearest, rich, host], dim=-1)
        flat_feat = self.flat_mlp(flat)
        return torch.cat([grid_feat, flat_feat], dim=-1)


def build_role_encoder(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererRoleEncoder(config)
    raise NotImplementedError(f"role {role!r} encoder not built (M2+)")
