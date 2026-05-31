"""Vectorized (batched-over-envs) skill dynamics for the gatherer fast-sim.

vec_apply_skills advances B parallel SimWorld skill-dynamics at once with pure
numpy, producing per-env world mutations BYTE-IDENTICAL (position to allclose
atol=1e-5; log_alive / oak count exactly) to looping the scalar
aiutopia.sim.skills.apply_skill over B independent worlds. It replaces the per-env
scalar HARVEST/NAVIGATE walk loops -- the fast-trainer bottleneck (~70-85% of
per-iter time at B=512, dominated by HARVEST tick-by-tick _walk_into_reach over up
to 64 logs per env).

HARVEST is the path-dependent sequential greedy nearest-neighbor chain (the agent
walks to each harvested log, so the next nearest is from the NEW position): modeled
as a LOOP over the chain index (<= MAX_QUANTITY == 64 iters) batched across the
active HARVEST envs. Each chain-iter does a batched nearest-alive scan (argmin over
(M, n)), a batched CLOSED-FORM walk to the chosen log, and a batched kill+increment,
with an active mask that drops envs which finished their cap or ran out of in-range
logs. NAVIGATE is a single batched closed-form walk. DEPOSIT / SEARCH / WAIT / NOOP
have no world side-effect.

CLOSED-FORM WALK: the scalar _walk_into_reach moves straight toward a FIXED target
at constant WALK_PER_TICK, so direction is constant and the position after k full
steps is start + unit0 * k * WALK. k = max(0, ceil((dist0 - R)/WALK)) then a
vectorized +/-1 correction using the loop EXACT squared (HARVEST) / linear
(NAVIGATE) stopping test. Fuzzed vs the scalar loop over 2e4 arena-range pairs +
reach-boundary cases: max position diff ~8e-13 (<< parity atol 1e-5).

CLIPPED BITSET (parity-critical for reward): the scalar path computes only the
relevant axis clip per skill -- HARVEST sets bit 3 (scalar OOB) only; NAVIGATE sets
bits 0/1/2 (spatial OOB) only; others 0. This mirrors that selectivity and returns
the per-env popcount (n_clipped) so VecGathererSim reward stays byte-identical. A
non-oak target_class HARVEST still computes its scalar clip bit even though it
collects nothing.

IMPORT-LIGHT by contract: numpy + aiutopia.sim.world only.
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.world import SimWorld

WALK_PER_TICK = 4.3 / 20.0
HARVEST_REACH = 4.5
_HARVEST_REACH_SQ = HARVEST_REACH * HARVEST_REACH + 1e-3
_HARVEST_REACH_R = float(np.sqrt(_HARVEST_REACH_SQ))
MAX_SEARCH_RADIUS = 48.0
NAV_ARRIVAL = 1.0
MAX_NAV_RANGE = 32.0
NAV_VERT_RANGE = 8.0
MAX_QUANTITY = 64
OAK_LOG_TARGET_CLASS = 0

SKILL_NAVIGATE = 0
SKILL_HARVEST = 1

__all__ = ["vec_apply_skills"]


def _closed_form_walk(
    start: np.ndarray,
    target: np.ndarray,
    reach_r: float,
    reach_sq: float,
    *,
    squared_test: bool,
) -> np.ndarray:
    """Batched closed-form straight walk toward a FIXED target at WALK_PER_TICK.

    start/target: (M, 3) float64. Returns (M, 3) final position, bit-equal (~1e-13)
    to looping the scalar walk per row. squared_test mirrors the scalar stopping
    predicate: HARVEST uses (dist0 - k*WALK)**2 <= reach_sq; NAVIGATE uses
    dist0 - k*WALK <= reach_r (linear, no epsilon) with the per-step min(WALK,dist)
    clamp folded in as min(k*WALK, dist0).
    """
    delta = target - start
    dist0 = np.sqrt((delta * delta).sum(axis=1))

    moving = (dist0 * dist0 > reach_sq) if squared_test else (dist0 > reach_r)

    safe_dist = np.where(dist0 > 0.0, dist0, 1.0)
    k_est = np.ceil((dist0 - reach_r) / WALK_PER_TICK)
    k = np.where(moving, np.maximum(0.0, k_est).astype(np.int64), 0)

    rem = dist0 - k.astype(np.float64) * WALK_PER_TICK
    if squared_test:
        k = np.where(moving & (rem * rem > reach_sq), k + 1, k)
        rem_prev = dist0 - (k - 1).astype(np.float64) * WALK_PER_TICK
        k = np.where(moving & (k > 0) & (rem_prev * rem_prev <= reach_sq), k - 1, k)
    else:
        k = np.where(moving & (rem > reach_r), k + 1, k)
        rem_prev = dist0 - (k - 1).astype(np.float64) * WALK_PER_TICK
        k = np.where(moving & (k > 0) & (rem_prev <= reach_r), k - 1, k)

    advance = k.astype(np.float64) * WALK_PER_TICK
    if not squared_test:
        advance = np.minimum(advance, dist0)
    unit = delta / safe_dist[:, None]
    return start + unit * advance[:, None]


def _clip_scalar_oob(scalar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clamp scalar to [0,1]; return (clamped, oob_mask). Mirrors _clip_scalar."""
    oob = (scalar < 0.0) | (scalar > 1.0)
    return np.clip(scalar, 0.0, 1.0), oob


def _clip_spatial_popcount(spatial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clamp spatial to [-1,1]^3; return (clamped, popcount of clipped axes)."""
    oob_axes = (spatial < -1.0) | (spatial > 1.0)
    return np.clip(spatial, -1.0, 1.0), oob_axes.sum(axis=1)


def _harvest_chain(
    harv_idx: np.ndarray,
    cap: np.ndarray,
    pos: np.ndarray,
    alive: np.ndarray,
    logs: np.ndarray,
    oak: np.ndarray,
) -> None:
    """Batched sequential greedy nearest-neighbor harvest chain (in place).

    Mirrors skills._apply_harvest: while broken < cap, find nearest alive log
    within MAX_SEARCH_RADIUS of floor(pos), walk into reach of its block center
    (log + 0.5), kill it, oak += 1. An env leaves the active set when it hits its
    cap OR has no in-range alive log left. cap is parallel to harv_idx (length M).
    """
    M = harv_idx.shape[0]
    if M == 0:
        return
    sub_pos = pos[harv_idx]
    sub_alive = alive[harv_idx]
    sub_logs = logs[harv_idx]
    broken = np.zeros(M, dtype=np.int64)

    for _chain in range(int(MAX_QUANTITY)):
        active = broken < cap
        if not active.any():
            break
        origin = np.floor(sub_pos).astype(np.int64)
        d = sub_logs - origin[:, None, :]
        dist = np.sqrt((d.astype(np.float64) ** 2).sum(axis=2))
        in_range = sub_alive & (dist <= MAX_SEARCH_RADIUS)
        do_harvest = active & in_range.any(axis=1)
        if not do_harvest.any():
            break
        masked = np.where(in_range, dist, np.inf)
        idx = np.argmin(masked, axis=1)
        sel = np.nonzero(do_harvest)[0]
        sel_idx = idx[sel]
        sel_center = sub_logs[sel, sel_idx].astype(np.float64) + 0.5
        sub_pos[sel] = _closed_form_walk(
            sub_pos[sel], sel_center, _HARVEST_REACH_R, _HARVEST_REACH_SQ, squared_test=True
        )
        sub_alive[sel, sel_idx] = False
        broken[sel] += 1

    pos[harv_idx] = sub_pos
    alive[harv_idx] = sub_alive
    oak[harv_idx] += broken


def vec_apply_skills(
    worlds: list[SimWorld],
    skill_type: np.ndarray,
    target_class: np.ndarray,
    spatial: np.ndarray,
    scalar: np.ndarray,
) -> np.ndarray:
    """Advance B worlds one macro-action each, vectorized over B; mutate in place.

    worlds: list[SimWorld] (mutated: agent_pos, log_alive, inventory["oak_log"]).
    skill_type/target_class: (B,) int. spatial: (B,3) float (raw). scalar: (B,) or
    (B,k) float (raw; first col read). Returns n_clipped (B,) int -- popcount of the
    per-env clippedAxesBitset with the scalar path per-skill selectivity. tick is
    left untouched (owned by VecGathererSim, as in the scalar path).
    """
    B = len(worlds)
    skill_type = np.asarray(skill_type).reshape(-1).astype(np.int64)
    target_class = np.asarray(target_class).reshape(-1).astype(np.int64)
    spatial = np.asarray(spatial, dtype=np.float64).reshape(B, 3)
    scalar_col = np.asarray(scalar, dtype=np.float64).reshape(B, -1)[:, 0]

    pos = np.stack([w.agent_pos for w in worlds]).astype(np.float64)
    alive = np.stack([w.log_alive for w in worlds]).astype(bool)
    logs = np.stack([w.logs for w in worlds]).astype(np.int64)
    oak = np.array([int(w.inventory.get("oak_log", 0)) for w in worlds], dtype=np.int64)
    n_clipped = np.zeros(B, dtype=np.int64)

    is_harvest = skill_type == SKILL_HARVEST
    is_navigate = skill_type == SKILL_NAVIGATE

    if is_navigate.any():
        nav_idx = np.nonzero(is_navigate)[0]
        sp_clamped, sp_clip_pc = _clip_spatial_popcount(spatial[nav_idx])
        n_clipped[nav_idx] = sp_clip_pc
        origin = pos[nav_idx]
        target = origin + sp_clamped * np.array(
            [MAX_NAV_RANGE, NAV_VERT_RANGE, MAX_NAV_RANGE], dtype=np.float64
        )
        pos[nav_idx] = _closed_form_walk(
            origin, target, NAV_ARRIVAL, NAV_ARRIVAL, squared_test=False
        )

    if is_harvest.any():
        harv_idx = np.nonzero(is_harvest)[0]
        sc_clamped, sc_oob = _clip_scalar_oob(scalar_col[harv_idx])
        n_clipped[harv_idx] = sc_oob.astype(np.int64)
        cap = np.maximum(1, np.rint(sc_clamped * MAX_QUANTITY).astype(np.int64))
        oak_class = target_class[harv_idx] == OAK_LOG_TARGET_CLASS
        cap = np.where(oak_class, cap, 0)
        _harvest_chain(harv_idx, cap, pos, alive, logs, oak)

    for j, w in enumerate(worlds):
        w.agent_pos = pos[j]
        if is_harvest[j]:
            w.log_alive = alive[j].copy()
            w.inventory["oak_log"] = int(oak[j])

    return n_clipped
