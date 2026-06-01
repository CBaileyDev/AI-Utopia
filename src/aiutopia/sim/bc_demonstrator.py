"""NAVIGATE-then-HARVEST scripted demonstrator (the BC oracle).

Anti-basin design (Research/SEED1_HOLE_DIAGNOSIS.md): the PPO gatherer learned a
degenerate HARVEST-press policy and never NAVIGATE, because randomized layouts almost
always gift a topmost-in-reach trunk at spawn, so the (HARVEST-masked, must-navigate-
first) state was effectively absent from training. The existing scripted demonstrators
(scripts/p0_gate_proof.py) HARVEST-SPAM -- they lean on HarvestSkill's internal walk
and NEVER emit NAVIGATE -- so cloning them reproduces the same degeneracy. THIS oracle
is corrective: on a HARVEST-masked obs it emits a NAVIGATE that steps toward the
nearest visible trunk; on an unmasked obs it presses HARVEST. Cloning it seeds the
navigate-then-harvest behavior PPO could not bootstrap.

NAVIGATE geometry: the bearing comes from ``g_nearest_resources[:, 0]`` (the nearest
trunk the NET perceives, so the clone uses only obs the policy sees). The bearing's dy
points at the *topmost* log (~+3 up); chasing it would float the agent and not transfer
to the real NavigateSkill, so the VERTICAL component is zeroed and movement is purely
horizontal. One horizontal NAVIGATE of the full bearing distance (<= 16b, within the
real MAX_NAV_RANGE=32) closes onto the trunk and unmasks HARVEST -- verified to drive
forced-masked spawns to the 64-oak goal at rate 1.000 and the 3 fixed gate seeds 3/3.

Action contract (batched, B-leading) matches VecGathererSim.step + the GathererActorHead
decode: skill_type 0=NAVIGATE 1=HARVEST; target_class 0 (oak_log, the only arena match);
scalar_param 1.0 (HARVEST cap 64); spatial_param raw in [-1,1]^3 applied by NavigateSkill
as origin + raw*[32, 8, 32]. comm fields are inert, present only for the contract.

IMPORT-LIGHT: numpy + the sim skill enum only -- never torch / py4j / aiutopia.env.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aiutopia.sim.skills import SKILL_HARVEST, SKILL_NAVIGATE

__all__ = ["demonstrate"]

_MAX_NAV_RANGE = 32.0  # NavigateSkill horizontal scale
_SCAN_RADIUS = 16.0  # g_nearest_resources dx/dz scaling
_EXPLORE_STEP_BLOCKS = 3.0  # default +x nudge when no trunk is perceived


def demonstrate(obs: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Batched oracle: unmasked -> HARVEST; masked -> NAVIGATE toward nearest trunk."""
    skill_mask = np.asarray(obs["action_mask"]["skill_type"])
    B = skill_mask.shape[0]
    harvest_ok = skill_mask[:, SKILL_HARVEST] == 1

    nearest = np.asarray(obs["g_nearest_resources"], dtype=np.float64)[:, 0, :]
    valid = nearest[:, 5] > 0.5  # the [..,5]=1 validity flag (a trunk is perceived)
    dx = nearest[:, 0] * _SCAN_RADIUS  # block-delta toward nearest trunk
    dz = nearest[:, 2] * _SCAN_RADIUS  # dy ignored (points at topmost log)

    spatial = np.zeros((B, 3), dtype=np.float32)
    spatial[:, 0] = (dx / _MAX_NAV_RANGE).astype(np.float32)
    spatial[:, 2] = (dz / _MAX_NAV_RANGE).astype(np.float32)

    # Fallback explore move when NO trunk is visible (rare: under force_masked the
    # trunk stays inside the 16b perception window). Without it a no-trunk row emits a
    # zero-movement NAVIGATE that silently stalls; a fixed +x nudge keeps it searching.
    no_trunk = ~valid
    if no_trunk.any():
        spatial[no_trunk, 0] = np.float32(_EXPLORE_STEP_BLOCKS / _MAX_NAV_RANGE)
        spatial[no_trunk, 2] = np.float32(0.0)

    skill_type = np.where(harvest_ok, SKILL_HARVEST, SKILL_NAVIGATE).astype(np.int64)
    return {
        "skill_type": skill_type,
        "target_class": np.zeros(B, dtype=np.int64),
        "spatial_param": spatial,
        "scalar_param": np.ones((B, 1), dtype=np.float32),
        "comm_payload": np.zeros((B, 128), dtype=np.float32),
        "should_broadcast": np.zeros(B, dtype=np.int64),
        "comm_target_mask": np.zeros((B, 4), dtype=np.int8),
    }
