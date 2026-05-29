"""Task 5 (lifecycle + termination parity) and Task 6 (scripted gate proof)
for the gatherer fast-sim env, AiUtopiaSimEnv.

Mirrors the PettingZoo-Parallel surface of AiUtopiaPettingZooEnv:
reset(seed) -> (obs, infos); step(action_dict) -> (obs, rew, term, trunc, info).
Termination parity with the wrapper: SUCCESS via _goal_success at {oak_log:64},
TRUNC on tick>=max_episode_ticks OR out-of-bounds.

IMPORT-LIGHT: drives only the sim package + env._embeds/spaces — no chroma /
py4j / torch / sentence_transformers.
"""

import numpy as np

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.train.scenario_runner import _gatherer_collected_64_oak_log


def _harvest(scalar=1 / 64):
    return {
        "gatherer_0": {
            "skill_type": 1,
            "target_class": 0,
            "spatial_param": np.zeros(3, np.float32),
            "scalar_param": np.array([scalar], np.float32),
            "comm_payload": np.zeros(128, np.float32),
            "should_broadcast": 0,
            "comm_target_mask": np.zeros(4, np.int8),
        }
    }


def test_reset_returns_obs_for_agent():
    env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "max_episode_ticks": 1000})
    obs, info = env.reset(seed=1)
    assert "gatherer_0" in obs


def test_success_terminates_at_64_oak_log():
    env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "max_episode_ticks": 1000})
    obs, _ = env.reset(seed=1)
    term = {"gatherer_0": False}
    steps = 0
    while not term["gatherer_0"] and steps < 70:
        obs, rew, term, trunc, info = env.step(_harvest(scalar=1 / 64))
        steps += 1
    assert term["gatherer_0"] is True
    assert info["gatherer_0"]["goal_success"] is True
    assert steps == 64  # 64 logs, cap=1 -> exactly 64 steps


def test_truncates_at_max_ticks():
    env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "max_episode_ticks": 5})
    env.reset(seed=1)
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(
            {
                "gatherer_0": {
                    "skill_type": 4,  # WAIT
                    "target_class": 0,
                    "spatial_param": np.zeros(3, np.float32),
                    "scalar_param": np.zeros(1, np.float32),
                    "comm_payload": np.zeros(128, np.float32),
                    "should_broadcast": 0,
                    "comm_target_mask": np.zeros(4, np.int8),
                }
            }
        )
    assert trunc["gatherer_0"] is True


# ───── Task 6: scripted gate proof in sim ─────
def test_scripted_policy_solves_gate_in_sim():
    """Sim analogue of scripts/p0_gate_proof.py: drive scripted HARVEST(cap=1)
    and assert scenario_runner's gate predicate passes on the final obs and the
    episode terminated with goal_success."""
    env = AiUtopiaSimEnv({"active_roles": ["gatherer"], "max_episode_ticks": 1000})
    obs, _ = env.reset(seed=1)
    final_obs = obs
    term = {"gatherer_0": False}
    info = {"gatherer_0": {}}
    for _ in range(120):  # 64 logs + slack
        obs, rew, term, trunc, info = env.step(_harvest(scalar=1 / 64))
        final_obs = obs
        if term["gatherer_0"] or trunc["gatherer_0"]:
            break
    assert _gatherer_collected_64_oak_log(final_obs) is True
    assert term["gatherer_0"] is True
    assert info["gatherer_0"]["goal_success"] is True
