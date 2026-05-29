# tests/unit/test_sim_reward_parity.py
import numpy as np

from aiutopia.env.reward import compute_reward_stage_1
from aiutopia.sim.obs_adapter import build_gatherer_obs
from aiutopia.sim.reward_adapter import step_reward
from aiutopia.sim.skills import apply_skill
from aiutopia.sim.world import SimWorld


# NOTE: this is a FORWARDING test (step_reward must just call compute_reward_stage_1),
# NOT a parity test — reward correctness rides entirely on obs/inventory faithfulness,
# which is covered by the golden-trace test in Task 3b. Don't mistake green here for fidelity.
def test_step_reward_forwards_to_compute_reward_stage_1():
    w = SimWorld()
    w.reset(seed=1)
    obs_prev = build_gatherer_obs(w)
    action = {
        "skill_type": 1,
        "target_class": 0,
        "spatial_param": np.zeros(3, np.float32),
        "scalar_param": np.array([1 / 64], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }
    w, comp = apply_skill(w, action)
    obs_curr = build_gatherer_obs(w)
    env_meta = {
        "died_this_tick": False,
        "n_clipped_param_axes": comp["clippedAxesBitset"].bit_count()
        if isinstance(comp["clippedAxesBitset"], int)
        else 0,
        "exploit_penalties": [],
    }
    got = step_reward(obs_prev, obs_curr, action, env_meta)
    want = compute_reward_stage_1(
        role="gatherer", obs_prev=obs_prev, obs_curr=obs_curr, action=action, env_meta=env_meta
    )
    assert got == want  # adapter must be a faithful pass-through
    assert got > 1.5  # +1 oak_log primary + PBRS, matches real probe
