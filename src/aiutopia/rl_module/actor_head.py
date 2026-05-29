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

import math
from typing import Any

import torch
from torch import nn

from aiutopia.env.spaces import (
    COMM_PAYLOAD_DIM,
    GOAL_EMBED_DIM,
    N_GATHERER_SKILLS,
    N_TARGET_CLASSES_PER_ROLE,
)

# Fix #4 — dead-comm masking constants.
#
# Per-dim diagonal-Gaussian (differential) entropy is
#     0.5 * log(2*pi*e*sigma^2)  =  0.5 + 0.5*log(2*pi) + log_std.
# Choosing log_std so this is exactly 0 makes each masked comm_payload dim
# contribute 0 to the summed entropy (and, being a fixed constant, 0 to the
# KL term and 0 gradient). This is preferable to driving sigma -> 0, which
# sends per-dim entropy to -inf and would blow up the entropy SUM the other
# way (a fresh NaN source).
LOG_STD_FOR_ZERO_ENTROPY: float = -0.5 * (1.0 + math.log(2.0 * math.pi))  # ≈ -1.4189

# should_broadcast (Discrete(2)) is forced near-deterministic with logits
# [+B, -B]. B is moderate (NOT inf) so logp / entropy stay numerically clean:
# entropy(softmax([20,-20])) ≈ 8e-17, gradient 0 (the logits are a constant).
_SHOULD_BROADCAST_LOGIT_MAGNITUDE: float = 20.0


# Per-action-key (kind, dim). Keys are listed alphabetically so the
# concatenated logits line up with tree.flatten(dict) ordering.
#   kind="categorical":  emits `dim` logits, paired with TorchCategorical(dim).
#   kind="gaussian":     emits 2*dim params (mean+std), paired with TorchDiagGaussian.
#   kind="multi_binary": emits 2*dim logits (a [2]*dim TorchMultiCategorical).
_GATHERER_HEAD_SLICES: dict[str, tuple[str, int]] = {
    "comm_payload": ("gaussian", COMM_PAYLOAD_DIM),  # 2 * 128 = 256
    "comm_target_mask": ("multi_binary", 4),  # 2 * 4   =   8
    "scalar_param": ("gaussian", 1),  # 2 * 1   =   2
    "should_broadcast": ("categorical", 2),  #             2
    "skill_type": ("categorical", N_GATHERER_SKILLS),  #             6
    "spatial_param": ("gaussian", 3),  # 2 * 3   =   6
    "target_class": ("categorical", N_TARGET_CLASSES_PER_ROLE),  #          64
}


def _output_size_for(kind: str, dim: int) -> int:
    if kind == "categorical":
        return dim
    # gaussian -> 2*dim (mean, log_std); multi_binary -> 2*dim (per-bit logits)
    return 2 * dim


def _slice_offsets() -> dict[str, tuple[int, int]]:
    """[start, end) of each action key inside the flat logits buffer.

    Derived from `_GATHERER_HEAD_SLICES` (alphabetical, matching
    `tree.flatten` of the Dict action space). Programmatic so a change to the
    action space's logit layout cannot silently desync the comm mask.
    """
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for key, (kind, dim) in _GATHERER_HEAD_SLICES.items():
        size = _output_size_for(kind, dim)
        offsets[key] = (cursor, cursor + size)
        cursor += size
    return offsets


def _comm_mask_slices() -> dict[str, tuple[int, int]]:
    """The flat-logit sub-slices the comm mask overwrites.

    comm_payload is a Gaussian (mean | log_std); we split it so the mean half
    is zeroed (deterministic action 0) and the log_std half is fixed to the
    zero-entropy constant. should_broadcast is the 2-logit categorical.
    """
    offs = _slice_offsets()
    cp_start, cp_end = offs["comm_payload"]
    mid = cp_start + COMM_PAYLOAD_DIM  # mean | log_std boundary
    return {
        "comm_payload_mean": (cp_start, mid),
        "comm_payload_logstd": (mid, cp_end),
        "should_broadcast": offs["should_broadcast"],
    }


def _build_comm_mask_const(output_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (keep_mask, const) for `logits = logits * keep_mask + const`.

    `keep_mask` is 1.0 everywhere except the masked comm slices (0.0 there);
    `const` is 0.0 everywhere except the masked slices, where it carries the
    fixed replacement values. The elementwise product both overwrites the
    masked values AND zeros the gradient flowing back into the corresponding
    output-layer rows — detach alone would leave the entropy VALUE
    uncontrolled (the raw log_std could still explode), so the overwrite is
    load-bearing, not just the stop-gradient.
    """
    keep = torch.ones(output_dim)
    const = torch.zeros(output_dim)
    sl = _comm_mask_slices()

    m0, m1 = sl["comm_payload_mean"]
    keep[m0:m1] = 0.0
    const[m0:m1] = 0.0  # deterministic mean = 0

    l0, l1 = sl["comm_payload_logstd"]
    keep[l0:l1] = 0.0
    const[l0:l1] = LOG_STD_FOR_ZERO_ENTROPY  # per-dim entropy = 0

    b0, b1 = sl["should_broadcast"]
    keep[b0:b1] = 0.0
    const[b0] = _SHOULD_BROADCAST_LOGIT_MAGNITUDE  # class 0 logit (+B)
    const[b0 + 1] = -_SHOULD_BROADCAST_LOGIT_MAGNITUDE  # class 1 logit (-B)

    return keep, const


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
    OUTPUT_DIM = sum(_output_size_for(kind, dim) for kind, dim in _GATHERER_HEAD_SLICES.values())

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

        # Fix #4: dead-comm masking. Reversible + config-controlled. Default
        # OFF so absent-key behavior is byte-identical to the legacy net (M2+
        # MARL, which reuses this role with LIVE comm, is thus safe-by-default;
        # M1B turns it ON explicitly in `m1_gatherer_config`).
        self.mask_comm: bool = bool(config.get("mask_comm", False))
        if self.mask_comm:
            keep, const = _build_comm_mask_const(self.OUTPUT_DIM)
            # Non-persistent: derived constants, not learned params — kept out
            # of the state_dict so checkpoints stay byte-identical across the
            # flag and the buffers follow the module under `.to(device)`.
            self.register_buffer("_comm_keep_mask", keep, persistent=False)
            self.register_buffer("_comm_const", const, persistent=False)

    def forward(self, hidden, goal_embedding):
        x = torch.cat([hidden, goal_embedding], dim=-1)
        logits = self.net(x)
        if self.mask_comm:
            # logits * keep + const: overwrites the comm slices with fixed
            # values AND zeros gradient into the producing rows. The masked
            # comm sub-distributions become constant -> ~0 entropy, ~0 KL
            # (identical at rollout and train time), ~0 gradient. Action-space
            # SHAPE is unchanged; the policy still EMITS comm_payload (zeroed)
            # and should_broadcast so the env contract is stable.
            logits = logits * self._comm_keep_mask + self._comm_const
        return logits


def build_actor_head(role: str, config: dict[str, Any]) -> nn.Module:
    if role == "gatherer":
        return GathererActorHead(config)
    raise NotImplementedError(f"actor head for {role!r} not built (M2+)")
