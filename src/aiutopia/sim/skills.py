"""Macro-skill dynamics for the gatherer fast-sim.

``apply_skill(world, action) -> (world, skill_completion)`` advances ``world``
by one *env step* (one macro-action), mutating it in place and returning it
together with a ``skill_completion`` dict shaped like the real Java side's
``CompletionEvent`` (MotorBridge.java): ``resultCode`` (str, one of
``SkillResult.name()`` -- COMPLETED / FAILED_TIMEOUT / IMMEDIATE_FAILURE /
RUNNING / ABORTED), ``failureReason`` (str), ``clippedAxesBitset`` (native int,
bits 0=spatial.dx 1=spatial.dy 2=spatial.dz 3=scalar).

Skill dynamics mirror ``fabric_mod/.../bridge/skill/*.java``:

- HARVEST (skill_type=1, HarvestSkill.java): scan within ``MAX_SEARCH_RADIUS``
  for the nearest alive log, walk toward it at ``WALK_PER_TICK`` per simulated
  tick until within ``HARVEST_REACH``, break it (``log_alive[i]=False``,
  ``inventory["oak_log"] += 1``); repeat up to ``cap=max(1, round(scalar*64))``
  blocks or until none remain. brokenCount==0 with no logs in range ->
  IMMEDIATE_FAILURE; otherwise COMPLETED.
- NAVIGATE (0, NavigateSkill.java): walk toward
  ``origin + spatial_param*[MAX_NAV_RANGE, 8, MAX_NAV_RANGE]`` until within
  ``NAV_ARRIVAL`` -> COMPLETED.
- DEPOSIT_CHEST (2): no chest in the M1B arena -> COMPLETED no-op (Phase A).
- SEARCH (3): no world side-effect -> COMPLETED.
- WAIT (4): no-op -> COMPLETED.
- NOOP_BROADCAST (5): no-op -> COMPLETED.

IMPORT-LIGHT by contract: this module imports only stdlib + numpy + the sibling
``aiutopia.sim.world`` -- never chromadb / py4j / torch / sentence_transformers,
and nothing from ``aiutopia.env`` (verified by the focused test and
``py -3.11 -c "import aiutopia.sim.skills"``).

CROSS-TASK NOTE (Task 5): ``apply_skill`` does NOT touch ``world.tick`` -- the
simulated walk-ticks consumed while approaching a log are an internal harvest
detail and must NOT inflate the env-step counter. ``world.tick`` is the
env-step counter and is owned/incremented by the SimEnv (Task 5), once per
``step``. Folding walk-ticks in here would push the 64-harvest success episode
past ``max_episode_ticks`` and break Task 5's termination-parity test.

FIDELITY NOTE (residual risk, deferred to Task 3b): on HARVEST we advance
``agent_pos`` to (just shy of) the harvested log -- the plan's "last harvested
log position" approximation. Real MC stops at vanilla AABB collision
(~0.8-1.5 blocks from the log center), not on the log. Because
``floor(agent_pos)`` becomes the obs-grid origin, this small offset is the
golden-trace tuning surface and is NOT pinned by any Task-2 test.
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.world import SimWorld

# ---------------------------------------------------------------------------
# Constants (mirror the Java skills exactly).
# ---------------------------------------------------------------------------
WALK_PER_TICK = 4.3 / 20.0  # 0.215 b/tick (HarvestSkill/NavigateSkill)
HARVEST_REACH = 4.5  # HarvestSkill.REACH_RADIUS (N16b)
MAX_SEARCH_RADIUS = 16.0  # HarvestSkill.MAX_SEARCH_RADIUS
NAV_ARRIVAL = 1.0  # NavigateSkill.ARRIVAL_RADIUS
MAX_NAV_RANGE = 32.0  # NavigateSkill.MAX_NAV_RANGE
NAV_VERT_RANGE = 8.0  # NavigateSkill vertical multiplier
MAX_QUANTITY = 64  # HarvestSkill.MAX_QUANTITY

# Generous walk-tick budget so a single harvest can always reach an in-range
# log; mirrors the spirit of the Java timeout (the agent walks ~16 blocks at
# 0.215 b/tick ~= 75 ticks before reaching the farthest in-range log).
_WALK_TICK_BUDGET = 2000

# Skill enum (Discrete(6)) -- confirmed against env/spaces.py + wrapper.py:
#   0=navigate 1=harvest 2=deposit_chest 3=search 4=wait 5=noop_broadcast
SKILL_NAVIGATE = 0
SKILL_HARVEST = 1
SKILL_DEPOSIT_CHEST = 2
SKILL_SEARCH = 3
SKILL_WAIT = 4
SKILL_NOOP_BROADCAST = 5


def _as_scalar(value, default: float) -> float:
    """Read scalar_param as a bare number or a 1-element array (R15 / readScalar)."""
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[0])


def _clip_scalar(scalar: float, clipped: int) -> tuple[float, int]:
    """Clamp scalar to [0,1]; set bit 3 of the clipped bitset if out of range."""
    if scalar < 0.0 or scalar > 1.0:
        clipped |= 0b1000
        scalar = max(0.0, min(1.0, scalar))
    return scalar, clipped


def _clip_spatial(action) -> tuple[np.ndarray, int]:
    """Clamp spatial_param to [-1,1]^3; set bits 0/1/2 for any clipped axis."""
    raw = np.asarray(
        action.get("spatial_param", np.zeros(3, dtype=np.float64)), dtype=np.float64
    ).reshape(-1)
    if raw.size < 3:
        raw = np.zeros(3, dtype=np.float64)
    clipped = 0
    out = raw[:3].copy()
    for i in range(3):
        if out[i] < -1.0 or out[i] > 1.0:
            clipped |= 1 << i
            out[i] = max(-1.0, min(1.0, out[i]))
    return out, clipped


def _completion(result_code: str, failure_reason: str = "", clipped: int = 0) -> dict:
    return {
        "resultCode": result_code,
        "failureReason": failure_reason,
        "clippedAxesBitset": int(clipped),
    }


def _nearest_alive_log(world: SimWorld) -> int:
    """Index of the nearest alive log within MAX_SEARCH_RADIUS, or -1.

    Mirrors HarvestSkill.findNearest/scanShell: Euclidean nearest within the
    16-block scan radius. All logs are flat at y=66, so the ground-preference
    two-pass scan (dy in [-2,+1] then full range) collapses to a plain nearest
    -- every log sits at dy=+1 relative to the agent's feet (y=65), inside the
    preferred band. We measure from the agent's BlockPos (floor) like the Java
    side, which scans integer block offsets from ``agent.getBlockPos()``.
    """
    alive = world.log_alive
    if not alive.any():
        return -1
    origin = np.floor(world.agent_pos).astype(np.int64)
    logs = world.logs
    d = logs - origin  # (64, 3)
    dist_sq = (d.astype(np.float64) ** 2).sum(axis=1)
    dist = np.sqrt(dist_sq)
    in_range = alive & (dist <= MAX_SEARCH_RADIUS)
    if not in_range.any():
        return -1
    masked = np.where(in_range, dist, np.inf)
    return int(np.argmin(masked))


def _walk_into_reach(world: SimWorld, target_center: np.ndarray) -> None:
    """Walk agent_pos toward target_center at WALK_PER_TICK until within reach.

    Mirrors the HarvestSkill move loop: full-speed steps along the normalized
    direction; stop when within HARVEST_REACH (squared, +1e-3 epsilon like the
    Java code). Does NOT advance ``world.tick`` (env-step counter is owned by
    the SimEnv).
    """
    reach_sq = HARVEST_REACH * HARVEST_REACH + 1e-3
    for _ in range(_WALK_TICK_BUDGET):
        delta = target_center - world.agent_pos
        if float(delta @ delta) <= reach_sq:
            return
        dist = float(np.sqrt(delta @ delta))
        direction = delta / dist
        world.agent_pos = world.agent_pos + direction * WALK_PER_TICK


def _apply_harvest(world: SimWorld, action) -> dict:
    scalar = _as_scalar(action.get("scalar_param"), 1.0 / MAX_QUANTITY)
    scalar, clipped = _clip_scalar(scalar, 0)
    cap = max(1, int(round(scalar * MAX_QUANTITY)))

    broken = 0
    while broken < cap:
        idx = _nearest_alive_log(world)
        if idx < 0:
            # No reachable log: mirror HarvestSkill -- COMPLETED if we already
            # broke at least one this dispatch, else IMMEDIATE_FAILURE.
            if broken > 0:
                break
            return _completion(
                "IMMEDIATE_FAILURE",
                f"no 'oak_log' within {MAX_SEARCH_RADIUS} blocks",
                clipped,
            )
        # Block center is (x+0.5, y+0.5, z+0.5) (Vec3d.ofCenter in the Java).
        target_center = world.logs[idx].astype(np.float64) + 0.5
        _walk_into_reach(world, target_center)
        world.log_alive[idx] = False
        world.inventory["oak_log"] = world.inventory.get("oak_log", 0) + 1
        broken += 1

    return _completion("COMPLETED", "", clipped)


def _apply_navigate(world: SimWorld, action) -> dict:
    raw, clipped = _clip_spatial(action)
    origin = world.agent_pos.copy()
    target = origin + raw * np.array(
        [MAX_NAV_RANGE, NAV_VERT_RANGE, MAX_NAV_RANGE], dtype=np.float64
    )
    for _ in range(_WALK_TICK_BUDGET):
        delta = target - world.agent_pos
        dist = float(np.sqrt(delta @ delta))
        if dist <= NAV_ARRIVAL:
            break
        direction = delta / dist
        step = min(WALK_PER_TICK, dist)
        world.agent_pos = world.agent_pos + direction * step
    return _completion("COMPLETED", "", clipped)


def apply_skill(world: SimWorld, action) -> tuple[SimWorld, dict]:
    """Advance ``world`` by one macro-action; return ``(world, skill_completion)``.

    ``world`` is mutated in place and also returned (callers rebind
    ``w, comp = apply_skill(w, action)``). ``world.tick`` is intentionally left
    untouched -- it is the env-step counter owned by the SimEnv (Task 5).
    """
    skill_type = int(np.asarray(action.get("skill_type", SKILL_NOOP_BROADCAST)).item())

    if skill_type == SKILL_HARVEST:
        completion = _apply_harvest(world, action)
    elif skill_type == SKILL_NAVIGATE:
        completion = _apply_navigate(world, action)
    elif skill_type == SKILL_DEPOSIT_CHEST:
        # No chest in the M1B arena -> COMPLETED no-op for Phase A (the Java
        # DepositChestSkill would IMMEDIATE_FAILURE on "no chest", but the sim
        # arena never contains one and the plan specifies a COMPLETED no-op).
        completion = _completion("COMPLETED")
    elif skill_type in (SKILL_SEARCH, SKILL_WAIT):
        completion = _completion("COMPLETED")
    else:  # SKILL_NOOP_BROADCAST (5) and any unknown skill id
        completion = _completion("COMPLETED")

    return world, completion
