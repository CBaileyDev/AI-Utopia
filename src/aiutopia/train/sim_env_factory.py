"""Env factory for the FAST-SIM gatherer backend (Phase B).

Sibling to ``env_factory.py`` but DELIBERATELY separate: ``env_factory.py``
imports ``AiUtopiaPettingZooEnv`` at module top, which drags in chromadb /
py4j / sentence-transformers. The whole point of the sim is to train with NONE
of that loaded, so the sim factory imports ONLY the import-light
``AiUtopiaSimEnv``. Wrapping in ``ParallelPettingZooEnv`` is identical to the
real path (``make_aiutopia_env_wrapped``) — the sim env subclasses
``pettingzoo.ParallelEnv`` so the wrapper treats it the same way.
"""
from __future__ import annotations

from typing import Any

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from aiutopia.sim.sim_env import AiUtopiaSimEnv


def make_aiutopia_sim_env(env_config: dict[str, Any]) -> AiUtopiaSimEnv:
    """Raw sim env factory; useful for non-RLlib consumers."""
    return AiUtopiaSimEnv(env_config)


def make_aiutopia_sim_env_wrapped(env_config: dict[str, Any]) -> ParallelPettingZooEnv:
    """RLlib-compatible factory: wraps the PettingZoo-Parallel sim env."""
    return ParallelPettingZooEnv(AiUtopiaSimEnv(env_config))
