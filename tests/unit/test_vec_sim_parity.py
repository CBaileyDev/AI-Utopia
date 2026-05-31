"""Per-env parity: VecGathererSim (batched) must match the scalar AiUtopiaSimEnv.

The lean no-Ray PPO loop steps B gatherer envs at once with pure numpy. sim<->real
parity is sacred, so the hard acceptance gate is that EVERY per-env result of
VecGathererSim is byte-identical (np.array_equal / allclose atol=1e-5) to a scalar
``AiUtopiaSimEnv`` seeded the same way, stepped through the SAME action sequence.

We compare the VANILLA gatherer path only (no decision_core, no scout, no shaping
flags) — that is the production M1 gate path the trainer learns on.

Action-sequence design (see advisor note): a full-cap HARVEST (scalar 1.0) clears
all 64 trees-arena logs in ONE step -> oak=64 -> goal_success -> term at step 1, so
we could not run 40 steps. The dynamics-parity sequence therefore uses cap-1
HARVESTs (scalar ~0.0156) interleaved with NAVIGATE / SEARCH / WAIT so the bag
stays < 64 and nothing terminates mid-run.
"""

from __future__ import annotations

import time

import numpy as np

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.vec_sim import VecGathererSim

_AGENT = "gatherer_0"
_FIELDS_ALLCLOSE = (
    "position",
    "g_resource_grid",
    "g_nearest_resources",
    "g_richness_score",
)
_MAX_TICKS = 300


def _scalar_env(seed: int) -> tuple[AiUtopiaSimEnv, dict]:
    env = AiUtopiaSimEnv(
        {
            "active_roles": ["gatherer"],
            "randomize_layout": False,
            "arena_mode": "trees",
            "max_episode_ticks": _MAX_TICKS,
        }
    )
    obs, _ = env.reset(seed=seed)
    return env, obs[_AGENT]


def _cap1_scalar() -> float:
    """Scalar_param that yields cap = max(1, round(s*64)) == 1 (one log per HARVEST)."""
    return 1.0 / 64.0  # round(1.0) == 1


def _action_sequence(T: int) -> list[dict]:
    """A mixed, NON-terminating action plan (cap-1 HARVESTs + NAV/SEARCH/WAIT)."""
    cap1 = _cap1_scalar()
    plan = [
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST one log
        (0, [0.3, 0.0, -0.2], 0.0),  # NAVIGATE
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST
        (3, [0.0, 0.0, 0.0], 0.0),  # SEARCH (noop)
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST
        (0, [-0.5, 0.1, 0.4], 0.0),  # NAVIGATE other way
        (4, [0.0, 0.0, 0.0], 0.0),  # WAIT (noop)
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST
        # HARVEST with OUT-OF-RANGE scalar (-0.5) -> _clip_scalar sets bit 3 ->
        # n_clipped=1 (exercises the r_clip reward sub-term) and clamps scalar to 0
        # so cap = max(1, round(0)) = 1: one log, no termination, no OOB walk.
        (1, [0.0, 0.0, 0.0], -0.5),
        (5, [0.0, 0.0, 0.0], 0.0),  # NOOP_BROADCAST (noop)
    ]
    seq: list[dict] = []
    for t in range(T):
        st, sp, sc = plan[t % len(plan)]
        seq.append(
            {
                "skill_type": int(st),
                "target_class": 0,
                "spatial_param": np.asarray(sp, dtype=np.float32),
                "scalar_param": np.asarray([sc], dtype=np.float32),
                "should_broadcast": 0,
                "comm_payload": np.zeros(1, dtype=np.float32),
                "comm_target_mask": np.zeros(4, dtype=np.int8),
            }
        )
    return seq


def _batched_actions(template: dict, B: int) -> dict[str, np.ndarray]:
    return {
        "skill_type": np.full(B, template["skill_type"], dtype=np.int64),
        "target_class": np.full(B, template["target_class"], dtype=np.int64),
        "spatial_param": np.tile(template["spatial_param"], (B, 1)),
        "scalar_param": np.tile(template["scalar_param"], (B, 1)),
        "should_broadcast": np.zeros(B, dtype=np.int64),
        "comm_payload": np.zeros((B, 1), dtype=np.float32),
        "comm_target_mask": np.zeros((B, 4), dtype=np.int8),
    }


def _assert_obs_equal(batched: dict, i: int, scalar_obs: dict, step: int) -> None:
    for key in _FIELDS_ALLCLOSE:
        bv = batched[key][i]
        sv = np.asarray(scalar_obs[key]).reshape(np.shape(bv))
        assert np.allclose(bv, sv, atol=1e-5), (
            f"obs[{key}] mismatch env {i} step {step}: max|d|={np.max(np.abs(bv - sv))}"
        )
    bc = np.asarray(scalar_obs["inv_slot_counts"])
    assert np.array_equal(batched["inv_slot_counts"][i], bc), f"inv counts env {i} step {step}"
    bi = np.asarray(scalar_obs["inv_slot_item_ids"])
    assert np.array_equal(batched["inv_slot_item_ids"][i], bi), f"inv ids env {i} step {step}"
    for mk in ("skill_type", "target_per_skill", "comm_payload", "should_broadcast"):
        bm = batched["action_mask"][mk][i]
        sm = np.asarray(scalar_obs["action_mask"][mk])
        assert np.array_equal(bm, sm), f"mask {mk} env {i} step {step}"


def test_vec_sim_matches_scalar_per_env() -> None:
    """Each VecGathererSim env must equal a scalar AiUtopiaSimEnv step-for-step."""
    seeds = np.array([1, 2, 3, 7], dtype=np.int64)
    B = len(seeds)
    T = 40
    seq = _action_sequence(T)

    vec = VecGathererSim(num_envs=B, max_episode_ticks=_MAX_TICKS)
    vec_obs = vec.reset(seeds)
    scalars: list[AiUtopiaSimEnv] = []
    for s in seeds:
        env, obs0 = _scalar_env(int(s))
        scalars.append(env)
        _assert_obs_equal(vec_obs, len(scalars) - 1, obs0, step=-1)

    # Enforce the "4 envs x 40 steps" coverage claim: every (env, step) pair must
    # be compared. The sequence is DESIGNED non-terminating (cap-1, bag < 64, no
    # OOB), so we also assert no env ever terminates/truncates mid-run — if it did,
    # the scalar env would prune the agent and silently skip later comparisons.
    compared = 0
    for t in range(T):
        template = seq[t]
        b_actions = _batched_actions(template, B)
        v_obs, v_rew, v_term, v_trunc = vec.step(b_actions)

        for i in range(B):
            env = scalars[i]
            assert env.agents, f"scalar env {i} pruned before step {t} (sequence must not end)"
            so, sr, st_, str_, _ = env.step({_AGENT: template})
            s_rew = sr[_AGENT]
            s_term = st_[_AGENT]
            s_trunc = str_[_AGENT]

            assert np.allclose(v_rew[i], s_rew, atol=1e-5), f"rew env {i} step {t}"
            assert bool(v_term[i]) == bool(s_term), f"term env {i} step {t}"
            assert bool(v_trunc[i]) == bool(s_trunc), f"trunc env {i} step {t}"
            assert not (s_term or s_trunc), f"env {i} ended at step {t} (sequence must not end)"
            _assert_obs_equal(v_obs, i, so[_AGENT], step=t)
            compared += 1

    assert compared == B * T, f"only compared {compared}/{B * T} (env, step) pairs"


def test_vec_sim_autoreset_on_success() -> None:
    """A terminated env (64-oak goal) must return the FRESH reset obs for its row."""
    seeds = np.array([1, 2], dtype=np.int64)
    B = len(seeds)
    vec = VecGathererSim(num_envs=B, max_episode_ticks=_MAX_TICKS)
    vec.reset(seeds)

    full = {
        "skill_type": np.full(B, 1, dtype=np.int64),  # HARVEST
        "target_class": np.zeros(B, dtype=np.int64),
        "spatial_param": np.zeros((B, 3), dtype=np.float32),
        "scalar_param": np.ones((B, 1), dtype=np.float32),  # cap 64 -> clears arena
        "should_broadcast": np.zeros(B, dtype=np.int64),
        "comm_payload": np.zeros((B, 1), dtype=np.float32),
        "comm_target_mask": np.zeros((B, 4), dtype=np.int8),
    }
    obs, _rew, term, _trunc = vec.step(full)
    assert bool(term[0]) and bool(term[1]), "both envs should reach the 64 goal"

    fresh = VecGathererSim(num_envs=B, max_episode_ticks=_MAX_TICKS)
    fresh_obs = fresh.reset(seeds)
    for i in range(B):
        assert np.array_equal(obs["position"][i], fresh_obs["position"][i]), f"pos env {i}"
        assert np.array_equal(obs["inv_slot_counts"][i], fresh_obs["inv_slot_counts"][i]), (
            f"inv env {i} (bag should be empty again)"
        )


def test_vec_sim_throughput_b512() -> None:
    """Report B=512 env-steps/s on the cap-1 HARVEST hot path (informational)."""
    B = 512
    vec = VecGathererSim(num_envs=B, max_episode_ticks=10_000)
    seeds = np.arange(1, B + 1, dtype=np.int64)
    vec.reset(seeds)
    cap1 = _cap1_scalar()
    actions = {
        "skill_type": np.full(B, 1, dtype=np.int64),
        "target_class": np.zeros(B, dtype=np.int64),
        "spatial_param": np.zeros((B, 3), dtype=np.float32),
        "scalar_param": np.full((B, 1), cap1, dtype=np.float32),
        "should_broadcast": np.zeros(B, dtype=np.int64),
        "comm_payload": np.zeros((B, 1), dtype=np.float32),
        "comm_target_mask": np.zeros((B, 4), dtype=np.int8),
    }
    n = 50
    vec.step(actions)  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        vec.step(actions)
    dt = time.perf_counter() - t0
    steps_per_s = n * B / dt
    print(f"\n[VecGathererSim B={B}] {steps_per_s:,.0f} env-steps/s ({n} steps/{dt:.3f}s)")
    assert steps_per_s > 0
