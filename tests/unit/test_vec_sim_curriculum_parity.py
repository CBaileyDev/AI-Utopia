"""Per-env parity for the TRAINING-only curriculum knobs on VecGathererSim.

The fast trainer's basin-escape curriculum (spawn_jitter / force_masked_spawn /
approach_shaping) was characterized on the SCALAR ``AiUtopiaSimEnv``. The
vectorized ``VecGathererSim`` must reproduce those scalar semantics per-env so
the fast trainer learns on the SAME curriculum. This is the acceptance gate:
VecGathererSim with a knob ON must equal the scalar env with the same knob ON,
step-for-step, on reward + the HARVEST-masked state + agent_pos.

Seed semantics (documented divergence): under ``randomize_layout=True`` the
scalar env IGNORES the reset seed and uses a GLOBAL ``_train_ep`` counter;
VecGathererSim carries PER-ENV seeds. We reconcile them by pinning the scalar's
``_train_ep`` so its effective layout seed equals the vec row's per-env seed.
WITHIN a fixed seed the jitter RNG, layout, and force-mask loop are bit-identical
(the vec spawn knobs run the SAME scalar arithmetic on a per-env ``SimWorld``).
Only the cross-episode seed schedule differs, which is documented in vec_sim.

We compare one masked-spawn episode over T non-terminating steps (cap-1 HARVEST +
NAVIGATE/SEARCH/WAIT, bag < 64, no autoreset in-window).
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.vec_sim import VecGathererSim

_AGENT = "gatherer_0"
_MAX_TICKS = 300


def _scalar_curriculum_env(seed: int, **knobs) -> tuple[AiUtopiaSimEnv, dict]:
    """Scalar curriculum env; pin ``_train_ep = seed-1`` so layout seed == seed."""
    env = AiUtopiaSimEnv(
        {
            "active_roles": ["gatherer"],
            "randomize_layout": True,
            "arena_mode": "trees",
            "max_episode_ticks": _MAX_TICKS,
            **knobs,
        }
    )
    env._train_ep = int(seed) - 1
    obs, _ = env.reset(seed=int(seed))
    return env, obs[_AGENT]


def _cap1() -> float:
    return 1.0 / 64.0


def _masked_action_sequence(T: int) -> list[dict]:
    """Mixed NON-terminating plan from a masked spawn (bag stays < 64)."""
    cap1 = _cap1()
    plan = [
        (0, [0.1, 0.0, 0.1], 0.0),  # NAVIGATE toward field
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST (one log if in reach)
        (3, [0.0, 0.0, 0.0], 0.0),  # SEARCH (noop)
        (0, [-0.1, 0.0, -0.1], 0.0),  # NAVIGATE back
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST
        (4, [0.0, 0.0, 0.0], 0.0),  # WAIT (noop)
        (0, [0.15, 0.0, -0.1], 0.0),  # NAVIGATE
        (1, [0.0, 0.0, 0.0], cap1),  # HARVEST
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


def _batched(template: dict, B: int) -> dict[str, np.ndarray]:
    return {
        "skill_type": np.full(B, template["skill_type"], dtype=np.int64),
        "target_class": np.full(B, template["target_class"], dtype=np.int64),
        "spatial_param": np.tile(template["spatial_param"], (B, 1)),
        "scalar_param": np.tile(template["scalar_param"], (B, 1)),
        "should_broadcast": np.zeros(B, dtype=np.int64),
        "comm_payload": np.zeros((B, 1), dtype=np.float32),
        "comm_target_mask": np.zeros((B, 4), dtype=np.int8),
    }


def _masked(skill_mask) -> bool:
    """HARVEST-masked == skill_type[1] == 0 (the perception/reach gate)."""
    return int(np.asarray(skill_mask).reshape(-1)[1]) == 0


def _run_knob_parity(seeds: np.ndarray, T: int, **knobs) -> None:
    B = len(seeds)
    seq = _masked_action_sequence(T)

    vec = VecGathererSim(num_envs=B, max_episode_ticks=_MAX_TICKS, randomize_layout=True, **knobs)
    vec_obs = vec.reset(seeds)

    scalars: list[AiUtopiaSimEnv] = []
    for i, s in enumerate(seeds):
        env, obs0 = _scalar_curriculum_env(int(s), **knobs)
        scalars.append(env)
        vp = vec_obs["position"][i]
        sp = obs0["position"]
        assert np.allclose(vp, sp, atol=1e-5), f"spawn pos env {i}: vec={vp} scalar={sp}"
        vm = _masked(vec_obs["action_mask"]["skill_type"][i])
        sm = _masked(obs0["action_mask"]["skill_type"])
        assert vm == sm, f"spawn mask env {i}: vec={vm} scalar={sm}"

    compared = 0
    for t in range(T):
        template = seq[t]
        v_obs, v_rew, v_term, v_trunc = vec.step(_batched(template, B))
        for i in range(B):
            env = scalars[i]
            assert env.agents, f"scalar env {i} pruned before step {t}"
            so, sr, st_, str_, _ = env.step({_AGENT: template})
            assert not (st_[_AGENT] or str_[_AGENT]), f"scalar env {i} ended at step {t}"
            assert not (bool(v_term[i]) or bool(v_trunc[i])), f"vec env {i} ended at step {t}"
            assert np.allclose(v_rew[i], sr[_AGENT], atol=1e-5), (
                f"rew env {i} step {t}: vec={v_rew[i]} scalar={sr[_AGENT]}"
            )
            vp = v_obs["position"][i]
            sp = so[_AGENT]["position"]
            assert np.allclose(vp, sp, atol=1e-5), f"pos env {i} step {t}"
            vm = _masked(v_obs["action_mask"]["skill_type"][i])
            sm = _masked(so[_AGENT]["action_mask"]["skill_type"])
            assert vm == sm, f"mask env {i} step {t}: vec={vm} scalar={sm}"
            compared += 1
    assert compared == B * T, f"only compared {compared}/{B * T}"


def test_curriculum_spawn_jitter_parity() -> None:
    """Spawn_jitter ON: vec per-env spawn + dynamics == scalar."""
    _run_knob_parity(np.array([1, 2, 3, 7], dtype=np.int64), T=24, spawn_jitter=4.0)


def test_curriculum_force_masked_spawn_parity() -> None:
    """Force_masked_spawn ON: pushed-out spawn + dynamics == scalar."""
    _run_knob_parity(np.array([1, 2, 3, 7], dtype=np.int64), T=24, force_masked_spawn=True)


def test_curriculum_approach_shaping_with_force_mask_parity() -> None:
    """Approach_shaping (paired with force-mask so it fires) == scalar."""
    _run_knob_parity(
        np.array([1, 2, 3, 7], dtype=np.int64),
        T=24,
        force_masked_spawn=True,
        approach_shaping=True,
    )


def test_curriculum_all_three_parity() -> None:
    """All three knobs ON together == scalar, per-env over T steps."""
    _run_knob_parity(
        np.array([1, 2, 3, 7], dtype=np.int64),
        T=24,
        spawn_jitter=4.0,
        force_masked_spawn=True,
        approach_shaping=True,
    )
