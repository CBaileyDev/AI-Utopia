"""Lock the decision-core oracle-ablation knobs added for the N23 characterization
(commit 7ad92c4): the HARVEST mask has three regimes and the bearing cue toggles.

These guard tonight's measurement infrastructure against regression:
  - DEFAULT (back-compat): reach-based mask — HARVEST valid only within REACH_RADIUS.
  - perception mask (decision_core): HARVEST valid when a trunk is VISIBLE in
    perception (the load-bearing crutch the ablation measured).
  - force-valid (harvest_mask_off): HARVEST unconditionally valid (ablates the mask).
  - resource_bearing_cue: g_hostiles_nearby[0] carries a unit direction + distance
    to the nearest alive log, else all-zeros.

IMPORT-LIGHT: drives only the sim package + env helpers (no chroma/py4j/torch).
"""

import numpy as np

from aiutopia.sim.obs_adapter import build_gatherer_obs
from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.world import SimWorld

# obs_adapter floors agent_pos -> origin; a clean integer pos keeps dx/dy/dz exact.
_AX, _AY, _AZ = 64.0, 65.0, -48.0


def _world_with_log_offsets(offsets):
    """A SimWorld with the agent at (_AX,_AY,_AZ) and one alive log per (dx,dy,dz)
    offset. Reset first to populate the non-spatial fields, then overwrite the
    spatial state so the geometry is exact and deterministic."""
    w = SimWorld()
    w.reset(1, arena_mode="clusters")
    w.agent_pos = np.array([_AX, _AY, _AZ], dtype=np.float64)
    w.logs = np.array(
        [[int(_AX) + dx, int(_AY) + dy, int(_AZ) + dz] for (dx, dy, dz) in offsets],
        dtype=np.int64,
    )
    w.log_alive = np.ones(len(offsets), dtype=bool)
    return w


def _harvest_bit(obs):
    return int(np.asarray(obs["action_mask"]["skill_type"])[1])  # skill 1 == HARVEST


# ── mask regimes ──────────────────────────────────────────────────────────────
def test_default_mask_is_reach_based_backcompat():
    """A log 10 blocks away is in PERCEPTION but NOT in reach (REACH_RADIUS=4.5).
    The default (no flags) reach-based mask must gate HARVEST OFF — proving the
    proven survival-forest path's mask is unchanged."""
    w = _world_with_log_offsets([(10, 0, 0)])
    obs = build_gatherer_obs(w)  # defaults: perception=False, force_valid=False
    assert _harvest_bit(obs) == 0


def test_reach_mask_on_when_in_reach():
    w = _world_with_log_offsets([(3, 0, 0)])  # dist 3 <= 4.5
    assert _harvest_bit(build_gatherer_obs(w)) == 1


def test_perception_mask_valid_when_visible_not_in_reach():
    """Same 10-block log: the perception mask (decision_core) gates HARVEST ON
    because the decision-core MINE walks to a visible target."""
    w = _world_with_log_offsets([(10, 0, 0)])
    assert _harvest_bit(build_gatherer_obs(w, harvest_mask_on_perception=True)) == 1


def test_force_valid_overrides_when_nothing_visible():
    """A log 20 blocks away is OUTSIDE the 16-block perception grid -> nothing
    visible. perception mask -> OFF; force_valid -> ON regardless (ablates the
    mask so the policy must learn not to mine when blind)."""
    w = _world_with_log_offsets([(20, 0, 0)])
    assert _harvest_bit(build_gatherer_obs(w, harvest_mask_on_perception=True)) == 0
    assert _harvest_bit(build_gatherer_obs(w, harvest_mask_force_valid=True)) == 1


# ── bearing cue ────────────────────────────────────────────────────────────────
def test_bearing_cue_off_is_all_zeros():
    w = _world_with_log_offsets([(10, 0, 0)])
    g = np.asarray(build_gatherer_obs(w)["g_hostiles_nearby"])
    assert not g.any()


def test_bearing_cue_on_points_at_nearest_log():
    """Cue row 0 = [unit dx, unit dz, dist/64, valid]; a log at +x gives +x unit."""
    w = _world_with_log_offsets([(10, 0, 0)])
    g = np.asarray(build_gatherer_obs(w, resource_bearing_cue=True)["g_hostiles_nearby"])
    assert g[0][3] == 1.0  # valid flag
    assert g[0][0] > 0.9  # unit-x toward +x log
    assert abs(g[0][1]) < 1e-6  # no z component
    assert 0.0 < g[0][2] <= 1.0  # normalized distance


# ── sim_env plumbing ────────────────────────────────────────────────────────────
def test_sim_env_knob_defaults_track_decision_core():
    plain = AiUtopiaSimEnv({"active_roles": ["gatherer"]})
    assert plain._harvest_perception_mask is False
    assert plain._harvest_mask_off is False

    dc = AiUtopiaSimEnv({"active_roles": ["gatherer"], "decision_core": True})
    assert dc._harvest_perception_mask is True  # defaults to decision_core
    assert dc._harvest_mask_off is False


def test_sim_env_mask_off_flag_threads_to_obs():
    """harvest_mask_off=True -> reset obs HARVEST bit is 1 even with no logs in
    reach (the env wires the knob through build_gatherer_obs)."""
    env = AiUtopiaSimEnv(
        {
            "active_roles": ["gatherer"],
            "decision_core": True,
            "arena_mode": "clusters",
            "harvest_mask_off": True,
        }
    )
    obs, _ = env.reset(seed=1)
    assert _harvest_bit(obs["gatherer_0"]) == 1
