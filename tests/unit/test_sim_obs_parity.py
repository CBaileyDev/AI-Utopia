"""Obs parity tests for the gatherer fast-sim.

NOTE (multi-task file): Task 3 (shape + range parity vs the declared space —
``test_obs_matches_contract_keys_and_is_contained``, ``test_resource_grid_lights_a_log_cell``,
``test_nearest_resources_top8_normalized``) is owned by a sibling agent and is
appended above/below this Task-3b block. This block is Task 3b ONLY: the
golden-trace fidelity gate that validates the sim against REAL Minecraft.

Task 3b — THE fidelity gate. Tasks 3/4/6 only check *internal validity*
(legal-in-space, function-equals-itself, sim-is-winnable). None verifies the sim
matches *Minecraft*. This test replays the exact scripted action sequence the
capture script used against the sim and asserts the sim reproduces the real-MC
obs field-by-field on the DYNAMIC set — the only test that catches byte-parity
bugs (channel-id map, ``oak_log -> 132`` inventory id, the ``dy = +1``
grid/nearest off-by-one) before they cost hours in Phase C.

The fixture ``tests/fixtures/gatherer_obs_trace_seed1.json`` requires a one-time
touch of a live Fabric instance (see ``scripts/capture_gatherer_obs_fixture.py``)
and is committed once captured. When ABSENT this test SKIPS cleanly (so the
suite stays green pre-capture, but the skip is visible).
"""

import importlib.util
import json
import pathlib

import numpy as np
import pytest

# Load the SCRIPTED_ACTIONS constant from the capture script as the single
# source of truth (so capture + replay can never drift) WITHOUT requiring
# `scripts/` to be an importable package (it is not on pythonpath, which is
# `src` only — see pyproject). A file-location import keeps the constant shared
# and the capture-script's heavy wrapper import lazy (inside its main()), so this
# stays import-light at collection time.
_CAPTURE_PY = (
    pathlib.Path(__file__).resolve().parents[2] / "scripts" / "capture_gatherer_obs_fixture.py"
)
_spec = importlib.util.spec_from_file_location("_capture_gatherer_obs_fixture", _CAPTURE_PY)
_capture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_capture)
_SCRIPTED_ACTIONS = _capture.SCRIPTED_ACTIONS

FIXTURE = pathlib.Path("tests/fixtures/gatherer_obs_trace_seed1.json")

# Fields the sim must match the real env on. NOTE: ``nearest_resource_distance``
# was dropped after capture confirmed the FINAL obs does not carry it
# (wrapper._decode_obs emits only space keys; the distance is consumed by the
# action-mask and discarded). Constant embeds are not asserted (the sim reuses
# the SAME _embeds functions so they match by construction).
DYNAMIC_FIELDS = [
    "position",
    "inv_slot_item_ids",
    "inv_slot_counts",
    "g_resource_grid",
    "g_nearest_resources",
    "g_richness_score",
]

# How many leading steps are DETERMINISTICALLY reproducible. The capture
# revealed (and the n14 probe corroborates by contradiction) that the real
# HARVEST is NON-DETERMINISTIC across back-to-back dispatches without movement:
# this fixture's HARVEST#1 collected 1 log but HARVEST#2/#3 collected 0, while a
# prior 6x-HARVEST probe collected 6/6. The idealized sim collects 1 per
# dispatch, so it cannot byte-match the post-harvest tail of THIS fixture. That
# divergence is a genuine measured SKILL-DYNAMICS fidelity gap (Phase-C work),
# NOT an obs-format bug. We therefore gate on the harvest-free prefix
# (post-reset + post-WAIT), which fully validates the obs FORMAT + the seed=1
# arena layout — i.e. the byte-parity bugs this test exists to catch
# (flat g_resource_grid, dy=+1 in g_nearest_resources, channel map, richness,
# seed-path parity). See the Phase-A report / spec §6 for the dynamics gap.
_DETERMINISTIC_PREFIX = 2  # trace[0]=post-reset, trace[1]=post-WAIT


def _assert_step(sim_obs, real_obs, step):
    for k in DYNAMIC_FIELDS:
        s = np.asarray(sim_obs[k], np.float64)
        r = np.asarray(real_obs[k], np.float64)
        assert s.shape == r.shape, f"step{step} {k}: shape {s.shape} != real {r.shape}"
        assert np.allclose(
            s, r, atol=1e-3
        ), f"step{step} {k}: sim != real (max |Δ|={np.abs(s - r).max():.4f})"


@pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="golden fixture not captured yet (needs a live instance once)",
)
def test_sim_obs_matches_real_golden_trace_prefix():
    """Obs-FORMAT fidelity gate on the deterministic, harvest-free prefix."""
    from aiutopia.sim.sim_env import AiUtopiaSimEnv

    trace = json.loads(FIXTURE.read_text())  # list of per-step real obs
    env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "max_episode_ticks": 1000})
    obs, _ = env.reset(seed=1)
    _assert_step(obs["gatherer_0"], trace[0], 0)
    for i, act in enumerate(_SCRIPTED_ACTIONS[: _DETERMINISTIC_PREFIX - 1]):
        obs, *_ = env.step({"gatherer_0": act})
        _assert_step(obs["gatherer_0"], trace[i + 1], i + 1)


# ── Task 3: internal-validity checks (sim obs conforms to the REAL obs FORMAT —
# the captured fixture, NOT the nominal space, which the real obs deviates from:
# g_resource_grid is flat (6144,), g_richness_score is a 0-d scalar). ──

# The 27 keys the real obs carries (== build_role_observation_space keys).
_REAL_OBS_KEYS = {
    "agent_uuid_embed",
    "role_one_hot",
    "tick_in_episode",
    "position",
    "velocity",
    "yaw_pitch",
    "health",
    "hunger",
    "saturation",
    "armor_value",
    "inv_slot_item_ids",
    "inv_slot_counts",
    "main_hand_item_id",
    "off_hand_item_id",
    "goal_embedding",
    "goal_ticks_left",
    "time_of_day",
    "weather",
    "biome_id",
    "light_level",
    "comm_payloads",
    "comm_metadata",
    "action_mask",
    "g_resource_grid",
    "g_nearest_resources",
    "g_richness_score",
    "g_hostiles_nearby",
}


def _fresh_obs():
    from aiutopia.sim.obs_adapter import build_gatherer_obs
    from aiutopia.sim.world import SimWorld

    w = SimWorld()
    w.reset(seed=1)
    return build_gatherer_obs(w)


def test_obs_keys_match_real_format():
    assert set(_fresh_obs().keys()) == _REAL_OBS_KEYS


def test_resource_grid_is_flat_6144_with_16_trunk_columns():
    g = np.asarray(_fresh_obs()["g_resource_grid"])
    assert g.shape == (6144,)  # FLAT, matching the real obs (not (32,32,6))
    assert set(np.unique(g)).issubset({0.0, 1.0})
    # N21 Inc2: 16 vertical trunks — each trunk's 4 logs share one (x,z) grid
    # cell, so exactly 16 cells light up (the flat (x,z) projection of the forest).
    assert int((g > 0).sum()) == 16


def test_nearest_resources_is_per_column_topmost_within_window():
    nr = np.asarray(_fresh_obs()["g_nearest_resources"])
    assert nr.shape == (8, 6)
    # N21 Inc2: matches GathererOverlayBuilder — ONE entry per (x,z) column at the
    # TOPMOST log within the ±3 vertical scan, NOT one row per stacked log. For
    # 4-tall trunks (Y66..69 from feet Y65) that topmost-in-window log is Y68 ->
    # dy=+3 -> dy/8 = 0.375 for every populated row (the Y69 crown is dy+4, above
    # the ±3 window, never seen — exactly as real MC reports it).
    populated = nr[np.any(nr != 0.0, axis=1)]
    assert len(populated) >= 1
    assert np.allclose(populated[:, 1], 0.375)
    cols = {(round(float(r[0]), 4), round(float(r[2]), 4)) for r in populated}
    assert len(cols) == len(populated), "nearest must be per-column, not per stacked log"
