"""Section 4.3 Per-role obs encoder. M1 ships gatherer; M2 adds explorer/farmer."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from aiutopia.rl_module.core_encoder import unflatten_role_obs


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
        # T21 fix #9 (now via shared helper): Ray 2.55's ConnectorV2 flattens
        # Dict obs values to (B, prod(shape)). unflatten_role_obs is idempotent
        # so this is safe for both unit-test 4D obs and live 2D-flat obs.
        grid = unflatten_role_obs(obs["g_resource_grid"], (32, 32, 6))
        grid = grid.permute(0, 3, 1, 2).contiguous()
        grid_feat = self.grid_conv(grid)

        nearest = obs["g_nearest_resources"].reshape(obs["g_nearest_resources"].shape[0], -1)

        rich = obs["g_richness_score"]
        if rich.ndim == 1:
            rich = rich.unsqueeze(-1)

        host = obs["g_hostiles_nearby"].reshape(obs["g_hostiles_nearby"].shape[0], -1)

        flat    = torch.cat([nearest, rich, host], dim=-1)
        flat_feat = self.flat_mlp(flat)
        return torch.cat([grid_feat, flat_feat], dim=-1)


class ExplorerRoleEncoder(nn.Module):
    """Explorer encoder: flatten richness + nearest_resources, no grid conv."""
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        # Explorer obs: position, richness_score, g_nearest_resources (8 logs with 6-d vectors)
        # Total: 1 + 8*6 = 49 dims
        flat_in = 1 + 8 * 6  # 49
        self.flat_mlp = nn.Sequential(
            nn.Linear(flat_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        """Flatten richness + nearest_resources into a single embedding."""
        rich = obs["g_richness_score"]
        if rich.ndim == 1:
            rich = rich.unsqueeze(-1)

        nearest = obs["g_nearest_resources"].reshape(obs["g_nearest_resources"].shape[0], -1)

        flat = torch.cat([rich, nearest], dim=-1)
        feat = self.flat_mlp(flat)
        return feat


class FarmerRoleEncoder(nn.Module):
    """Farmer encoder: ConvNet on f_crop_grid + flat MLP on ripeness/planted/harvested."""
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        # Crop grid: 32x32 spatial, 1 channel (crop age 0-8)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
        )
        # Scalar features: f_ripeness (1) + f_planted_count (1) + f_harvested_count (1)
        flat_in = 3
        self.flat_mlp = nn.Sequential(
            nn.Linear(flat_in, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, obs):
        """Encode f_crop_grid via ConvNet; ripeness/planted/harvested via MLP."""
        # f_crop_grid: reshape to (B, 1, 32, 32)
        grid = unflatten_role_obs(obs["f_crop_grid"], (32, 32))
        if grid.ndim == 2:  # (B, 1024) -> (B, 1, 32, 32)
            grid = grid.view(grid.shape[0], 1, 32, 32)
        grid_feat = self.grid_conv(grid)

        # f_ripeness, f_planted_count, f_harvested_count: scalar per batch
        ripeness = obs["f_ripeness"]
        if ripeness.ndim == 1:
            ripeness = ripeness.unsqueeze(-1)

        planted = obs["f_planted_count"]
        if planted.ndim == 1:
            planted = planted.unsqueeze(-1).float()

        harvested = obs["f_harvested_count"]
        if harvested.ndim == 1:
            harvested = harvested.unsqueeze(-1).float()

        # Normalize planted/harvested to [0, 1]
        planted_norm = torch.clamp(planted / 64.0, 0.0, 1.0)
        harvested_norm = torch.clamp(harvested / 64.0, 0.0, 1.0)

        flat = torch.cat([ripeness, planted_norm, harvested_norm], dim=-1)
        flat_feat = self.flat_mlp(flat)

        return torch.cat([grid_feat, flat_feat], dim=-1)


def build_role_encoder(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererRoleEncoder(config)
    elif role == "explorer":
        return ExplorerRoleEncoder(config)
    elif role == "farmer":
        return FarmerRoleEncoder(config)
    raise NotImplementedError(f"role {role!r} encoder not built")
