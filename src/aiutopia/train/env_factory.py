"""Env factory for Ray Tune. Wraps PettingZoo Parallel env for RLlib."""
from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from aiutopia.env.wrapper import AiUtopiaPettingZooEnv


def make_aiutopia_env(env_config: dict[str, Any]) -> AiUtopiaPettingZooEnv:
    """Raw env factory; useful for non-RLlib consumers."""
    return AiUtopiaPettingZooEnv(env_config)


def make_aiutopia_env_wrapped(env_config: dict[str, Any]) -> ParallelPettingZooEnv:
    """RLlib-compatible factory: wraps the PettingZoo Parallel env."""
    return ParallelPettingZooEnv(AiUtopiaPettingZooEnv(env_config))
