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
        # obs values may arrive as (B, H, W, C) or (B*T, H, W, C); same op works
        grid = obs["g_resource_grid"].permute(0, 3, 1, 2).contiguous()
        grid_feat = self.grid_conv(grid)
        nearest = obs["g_nearest_resources"].flatten(start_dim=1)
        rich    = obs["g_richness_score"]
        host    = obs["g_hostiles_nearby"].flatten(start_dim=1)
        flat    = torch.cat([nearest, rich, host], dim=-1)
        flat_feat = self.flat_mlp(flat)
        return torch.cat([grid_feat, flat_feat], dim=-1)


def build_role_encoder(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererRoleEncoder(config)
    raise NotImplementedError(f"role {role!r} encoder not built (M2+)")
