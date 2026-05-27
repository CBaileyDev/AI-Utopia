"""1-iteration PPO smoke test on a synthetic gatherer env. No Fabric needed.

Run:
  PYTHONPATH=src py -3.11 scripts/rllib_smoke.py

This catches RLlib integration bugs (e.g. _forward shape mismatches with the
actual Connector pipeline) before T21 burns hours on a real Fabric server.
Uses a synthetic PettingZoo env that returns the gatherer obs space sample()
so no Java/Fabric/Py4J is required.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import ray
from pettingzoo import ParallelEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from aiutopia.env.spaces import build_role_action_space, build_role_observation_space
from aiutopia.train.config import m1_gatherer_config, ENV_NAME


class _SyntheticGathererEnv(ParallelEnv):
    """Minimal PettingZoo Parallel env over the real gatherer obs/action space.

    Returns random obs samples and noisy rewards, truncating every 16 steps.
    No external dependencies — pure in-process.
    """

    metadata = {"render_modes": []}

    def __init__(self, render_mode=None):
        self.agents = ["gatherer_0"]
        self.possible_agents = list(self.agents)
        self._obs_space = build_role_observation_space("gatherer", stage=1)
        self._action_space = build_role_action_space("gatherer")
        self._step = 0

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self._step = 0
        return (
            {a: self._obs_space.sample() for a in self.agents},
            {a: {} for a in self.agents},
        )

    def step(self, actions):
        self._step += 1
        obs = {a: self._obs_space.sample() for a in self.agents}
        rew = {a: float(np.random.randn()) for a in self.agents}
        term = {a: False for a in self.agents}
        trunc = {a: self._step >= 16 for a in self.agents}
        info = {a: {} for a in self.agents}
        return obs, rew, term, trunc, info


def _make_synthetic_wrapped(env_config):
    return ParallelPettingZooEnv(_SyntheticGathererEnv())


def main() -> int:
    # Initialize Ray. local_mode was removed in Ray 2.55+; try it first for
    # back-compat with 2.40-2.54, then fall back to plain init.
    try:
        ray.init(local_mode=True, num_cpus=2, num_gpus=0, ignore_reinit_error=True)
    except (TypeError, RuntimeError):
        ray.init(num_cpus=2, num_gpus=0, ignore_reinit_error=True)

    try:
        # IMPORTANT: m1_gatherer_config() internally calls register_aiutopia_env(),
        # which registers ENV_NAME -> the real (Java/Py4J) factory. We must
        # OVERWRITE that registration AFTER the config is built so env runners
        # spawn the synthetic env instead. (Plan's "Re-register" comment.)
        cfg = m1_gatherer_config(num_env_runners=0, num_envs_per_env_runner=1)
        register_env(ENV_NAME, _make_synthetic_wrapped)

        # Override learner GPU allocation to 0 for CPU-only smoke.
        # num_learners=0 keeps the Learner in the driver process and avoids
        # spawning a remote Train worker (which on Windows would try to
        # init a libuv-backed torch process group and crash).
        cfg = cfg.learners(num_learners=0, num_gpus_per_learner=0)

        algo = cfg.build()
        result = algo.train()

        # Sanity: training executed and produced a result dict with metrics.
        assert isinstance(result, dict), f"expected dict result, got {type(result)}"
        assert "info" in result or "env_runners" in result, result

        print("RLLIB SMOKE OK")
        algo.stop()
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
