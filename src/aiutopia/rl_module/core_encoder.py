"""Section 4.3 CoreEncoder — universal core obs to 256-d feature.

Per the v2 review:
- vectorized batched flatten (no per-sample Python loop)
- Discrete keys go through nn.Embedding instead of float cast
- belongs to AiUtopiaRoleRLModule as an instance attribute, NOT a multi-module-spec entry
"""
# NOTE (v2 defect, revisit in M2): the v2 prose calls for nn.Embedding on
# Discrete keys (biome_id, weather), but this implementation casts them to
# float32 alongside everything else. Faithful to the v2 plan code body; the
# representational upgrade is deferred to M2.
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete, MultiBinary, Dict as DictSpace

from aiutopia.env.spaces import build_role_observation_space


# Universal core obs keys. Role-overlay keys (g_*, b_*, f_*, d_*) and
# `action_mask` are deliberately excluded; those belong to the per-role encoder.
_CORE_KEYS_FOR_FLATTEN = (
    "agent_uuid_embed", "role_one_hot", "tick_in_episode",
    "position", "velocity", "yaw_pitch",
    "health", "hunger", "saturation", "armor_value",
    "inv_slot_item_ids", "inv_slot_counts",
    "main_hand_item_id", "off_hand_item_id",
    "goal_embedding", "goal_ticks_left",
    "time_of_day", "weather", "biome_id", "light_level",
    "comm_payloads", "comm_metadata",
)


def core_obs_flat_dim() -> int:
    """Total dim of the concatenated core obs (before encoder)."""
    space = build_role_observation_space("gatherer", stage=1)
    total = 0
    for key in _CORE_KEYS_FOR_FLATTEN:
        sub = space.spaces[key]
        total += int(np.prod(sub.shape)) if sub.shape else 1
    return total


def unflatten_role_obs(t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Restore a flattened role-overlay obs tensor to its spatial shape.

    Ray 2.55's ConnectorV2 flattens Dict obs values to (B, prod(shape)) before
    passing them to the RLModule. Per-role encoders that need the spatial
    structure (conv2d on a grid, patch encoder for pixel) must call this to
    restore (B, *shape).

    Idempotent — if `t` already has the spatial shape, returns it unchanged.

    Examples:
        >>> grid = unflatten_role_obs(obs["g_resource_grid"], (32, 32, 6))
        >>> grid = grid.permute(0, 3, 1, 2).contiguous()       # (B, C, H, W)

        >>> patch = unflatten_role_obs(obs["b_pixel_patch"], (64, 64, 3))
        >>> patch = patch.permute(0, 3, 1, 2).contiguous()     # M2 builder
    """
    if t.ndim == 1 + len(shape):
        return t                                                # already (B, *shape)
    if t.ndim == 2 and t.shape[1] == int(np.prod(shape)):
        return t.reshape(t.shape[0], *shape)                    # flattened (B, prod)
    raise ValueError(
        f"unflatten_role_obs: cannot reconcile tensor shape {tuple(t.shape)} "
        f"with expected role obs shape (B, *{shape})"
    )


def flatten_core_obs_batched(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Vectorized flatten: dict of (B, ...) tensors -> (B, D) float32 tensor.

    All present batch dim; scalar Discretes show up as (B,) and get
    unsqueezed to (B, 1). Outputs stay on the input device.
    """
    parts: list[torch.Tensor] = []
    for key in _CORE_KEYS_FOR_FLATTEN:
        v = obs[key]
        if not torch.is_tensor(v):
            v = torch.as_tensor(np.asarray(v))
        if v.dtype != torch.float32:
            v = v.to(torch.float32)
        if v.ndim == 1:
            v = v.unsqueeze(-1)
        else:
            v = v.reshape(v.shape[0], -1)
        parts.append(v)
    return torch.cat(parts, dim=-1)


class CoreEncoderModule(nn.Module):
    """Universal core obs (flat, ~5k-d) -> 256-d feature.

    Instantiated as `self.core_encoder` on AiUtopiaRoleRLModule. For M1
    there's one policy, so no cross-policy weight sharing concern. M2
    will revisit if the builder/farmer benefit from sharing.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        hidden = config.get("core_hidden", [512, 256])
        in_dim = core_obs_flat_dim()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden[-1]

    def forward(self, core_flat: torch.Tensor) -> torch.Tensor:
        return self.net(core_flat)
