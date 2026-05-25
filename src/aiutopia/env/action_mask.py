"""§4.5 — Hard action mask for gatherer (M0).

Mask is computed in Python from the symbolic obs (Python is the only place
that sees both the obs and the role rules). Java sends raw world facts;
Python decides which actions are legal.

The mask shape MUST match the action_mask sub-dict declared in spaces.py."""
from __future__ import annotations

import numpy as np

from aiutopia.env.spaces import N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE

# Skill indices for gatherer (must match Java motor side):
NAVIGATE        = 0
HARVEST         = 1
DEPOSIT_CHEST   = 2
SEARCH          = 3
WAIT            = 4
NOOP_BROADCAST  = 5


def compute_gatherer_action_mask(obs_raw: dict) -> dict[str, np.ndarray]:
    """Build the action_mask sub-dict from a raw observation dict.

    `obs_raw` is the Python-side parse of the JSON blob from
    `Py4JEntryPoint.observationsAll()` — has at minimum:
      - inv_slot_counts: list[int] of length 36
      - target_chest_in_range: bool
      - target_resource_in_range: bool
      - health: float
    """
    skill_mask  = np.ones(N_GATHERER_SKILLS, dtype=np.int8)
    target_mask = np.ones((N_GATHERER_SKILLS, N_TARGET_CLASSES_PER_ROLE),
                          dtype=np.int8)

    inv_full = sum(obs_raw.get("inv_slot_counts", [0] * 36)) >= 36 * 64
    if inv_full:
        skill_mask[HARVEST] = 0

    if not obs_raw.get("target_chest_in_range", False):
        target_mask[DEPOSIT_CHEST, :] = 0
        skill_mask[DEPOSIT_CHEST]    = 0
    if not obs_raw.get("target_resource_in_range", False):
        target_mask[HARVEST, :] = 0
        if skill_mask[HARVEST] == 1:
            skill_mask[HARVEST] = 0

    # Fail-safe: if every skill is masked, allow WAIT (§4.5 guard).
    if not skill_mask.any():
        skill_mask[WAIT] = 1

    return {
        "skill_type":       skill_mask,
        "target_per_skill": target_mask,
        "comm_payload":     np.ones(1, dtype=np.int8),
        "should_broadcast": np.ones(2, dtype=np.int8),
    }
