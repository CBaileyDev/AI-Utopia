"""Section 4.3 ActorHead — per-role action distribution producer."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aiutopia.env.spaces import (
    COMM_PAYLOAD_DIM,
    GOAL_EMBED_DIM,
    N_GATHERER_SKILLS,
    N_TARGET_CLASSES_PER_ROLE,
)


_GATHERER_HEAD_SLICES = {
    "skill_type":       ("logits", N_GATHERER_SKILLS),
    "target_class":     ("logits", N_TARGET_CLASSES_PER_ROLE),
    "spatial_param":    ("gaussian", 3),
    "scalar_param":     ("gaussian", 1),
    "comm_payload":     ("gaussian", COMM_PAYLOAD_DIM),
    "should_broadcast": ("logits", 2),
    "comm_target_mask": ("logits", 4),
}


def _output_size_for(kind: str, dim: int) -> int:
    return dim if kind == "logits" else 2 * dim


class GathererActorHead(nn.Module):
    INPUT_DIM = 256 + GOAL_EMBED_DIM
    OUTPUT_DIM = sum(_output_size_for(kind, dim)
                      for kind, dim in _GATHERER_HEAD_SLICES.values())

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        actor_hidden = config.get("actor_hidden", [256])
        layers: list[nn.Module] = []
        prev = self.INPUT_DIM
        for h in actor_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, self.OUTPUT_DIM))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden, goal_embedding):
        x = torch.cat([hidden, goal_embedding], dim=-1)
        return self.net(x)


def build_actor_head(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererActorHead(config)
    raise NotImplementedError(f"actor head for {role!r} not built (M2+)")
