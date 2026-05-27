"""Section 4.3 ActorHead — per-role action distribution producer.

Slice order is **alphabetical** to match `tree.flatten` on the Dict action
space (gymnasium sorts Dict keys alphabetically). RLlib's
`TorchMultiDistribution.from_logits` splits the logits buffer in
`tree.flatten(input_lens)` order, so the actor head must emit logits in
that same order or the wrong slice flows into each child distribution.

MultiBinary(N) is modeled as TorchMultiCategorical with [2]*N categories →
2*N logits, not N. The original plan's `OUTPUT_DIM=340` missed this and is
corrected to 344 here.
"""
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


# Per-action-key (kind, dim). Keys are listed alphabetically so the
# concatenated logits line up with tree.flatten(dict) ordering.
#   kind="categorical":  emits `dim` logits, paired with TorchCategorical(dim).
#   kind="gaussian":     emits 2*dim params (mean+std), paired with TorchDiagGaussian.
#   kind="multi_binary": emits 2*dim logits (a [2]*dim TorchMultiCategorical).
_GATHERER_HEAD_SLICES: dict[str, tuple[str, int]] = {
    "comm_payload":     ("gaussian",     COMM_PAYLOAD_DIM),       # 2 * 128 = 256
    "comm_target_mask": ("multi_binary", 4),                       # 2 * 4   =   8
    "scalar_param":     ("gaussian",     1),                       # 2 * 1   =   2
    "should_broadcast": ("categorical",  2),                       #             2
    "skill_type":       ("categorical",  N_GATHERER_SKILLS),       #             6
    "spatial_param":    ("gaussian",     3),                       # 2 * 3   =   6
    "target_class":     ("categorical",  N_TARGET_CLASSES_PER_ROLE),  #          64
}


def _output_size_for(kind: str, dim: int) -> int:
    if kind == "categorical":
        return dim
    # gaussian -> 2*dim (mean, log_std); multi_binary -> 2*dim (per-bit logits)
    return 2 * dim


def gatherer_action_dist_config() -> tuple[dict, dict]:
    """Return (child_distribution_cls_struct, input_lens) for TorchMultiDistribution.

    Importing the Ray distribution classes is deferred to keep this module
    cheap to import in test-only contexts.
    """
    from ray.rllib.core.distribution.torch.torch_distribution import (
        TorchCategorical,
        TorchDiagGaussian,
        TorchMultiCategorical,
    )

    child_struct: dict = {}
    input_lens: dict = {}
    for key, (kind, dim) in _GATHERER_HEAD_SLICES.items():
        size = _output_size_for(kind, dim)
        input_lens[key] = size
        if kind == "categorical":
            child_struct[key] = TorchCategorical
        elif kind == "gaussian":
            child_struct[key] = TorchDiagGaussian
        elif kind == "multi_binary":
            # MultiBinary(N) -> N independent 2-way categoricals.
            child_struct[key] = TorchMultiCategorical.get_partial_dist_cls(
                input_lens=[2] * dim,
            )
        else:
            raise ValueError(f"unknown slice kind {kind!r} for {key!r}")
    return child_struct, input_lens


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
