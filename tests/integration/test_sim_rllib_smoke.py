"""End-to-end 1-iteration PPO smoke on the FAST-SIM gatherer env (no live MC).

This is the Phase-B cheap-path proof: the SAME RLModule / PPO stack that trains
against real Minecraft (``tests/integration/test_rllib_smoke.py``) runs one full
training iteration against ``AiUtopiaSimEnv`` — pure in-process NumPy sim, no
Fabric / Py4J / Chroma server. Because there is no live server it deliberately
runs in the DEFAULT suite (no ``@pytest.mark.integration``).

Mirrors the Windows/CPU-safe knobs from ``scripts/rllib_smoke.py``:
  - ``num_env_runners=0`` (local in-driver runner — the proven CPU path),
  - ``num_learners=0`` (learner in-driver, dodges the Windows libuv crash),
  - ``num_gpus_per_learner=0`` + ``ray.init(num_gpus=0)`` (CPU-only),
  - ``local_mode=True`` try/except fallback (kwarg removed in Ray 2.55+).

Unlike the real-MC smoke this asserts ``env_runners/episode_return_mean`` is
present and finite: sim episodes complete in a handful of steps (a single
HARVEST with scalar≈1 clears the 64-log ring → success-terminate; otherwise
truncation at ``max_episode_ticks=300``), so many episodes finish inside one
iteration and the smoothed mean populates (the real env's windowed mean reads
~0 early because almost no ~800s-each episodes have completed yet).
"""
from __future__ import annotations

import math

import pytest

ray = pytest.importorskip("ray")


def _episode_return_mean(result: dict) -> float | None:
    """Pull ``env_runners/episode_return_mean`` out of the result dict.

    Ray 2.55 nests it under ``result["env_runners"]``; older stacks may expose
    a flat ``env_runners/episode_return_mean`` key. Return None if absent.
    """
    er = result.get("env_runners")
    if isinstance(er, dict) and "episode_return_mean" in er:
        return er["episode_return_mean"]
    if "env_runners/episode_return_mean" in result:
        return result["env_runners/episode_return_mean"]
    return None


def test_sim_rllib_runs_one_iteration() -> None:
    from aiutopia.train.config import SIM_ENV_NAME, m1_gatherer_config

    # local_mode was removed in Ray 2.55+; try it (back-compat) then fall back.
    try:
        ray.init(local_mode=True, num_cpus=2, num_gpus=0, ignore_reinit_error=True)
    except (TypeError, RuntimeError):
        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

    algo = None
    try:
        cfg = m1_gatherer_config(
            backend="sim",
            num_env_runners=0,
            num_envs_per_env_runner=1,
        )
        # SAME Windows/CPU-safe learner override as scripts/rllib_smoke.py.
        cfg = cfg.learners(num_learners=0, num_gpus_per_learner=0)

        # The config must point at the sim env, not the real-MC env.
        assert cfg.env == SIM_ENV_NAME
        # Real-MC-only env_config keys must be dropped on the sim backend.
        assert "py4j_ports" not in cfg.env_config
        assert "tick_warp" not in cfg.env_config

        algo = cfg.build()
        result = algo.train()

        assert isinstance(result, dict), f"expected dict result, got {type(result)}"
        # The whole point: this stack produces env-runner episode metrics in sim.
        erm = _episode_return_mean(result)
        er = result.get("env_runners")
        er_keys = sorted(er.keys()) if isinstance(er, dict) else "n/a"
        assert erm is not None, (
            "env_runners/episode_return_mean missing; "
            f"result keys={sorted(result.keys())} env_runners keys={er_keys}"
        )
        assert math.isfinite(float(erm)), f"episode_return_mean not finite: {erm}"
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()
