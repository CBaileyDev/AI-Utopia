"""Confirm the BC checkpoint emits NAVIGATE then HARVEST on the canonical seed_1
gate scenario (AiUtopiaSimEnv, greedy mask-aware decode = the gate path)."""

from __future__ import annotations

import numpy as np
import torch
from fast_train import build_module
from ray.rllib.core import Columns

from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.train.scenario_runner import M1_SCENARIOS, _greedy_decode

device = "cuda" if torch.cuda.is_available() else "cpu"
mod = build_module(device)
sd = torch.load("weights/bc_gatherer.pt", map_location=device)
mod.load_state_dict(sd, strict=False)
mod.eval()

sc = next(s for s in M1_SCENARIOS if s.seed == 1)
env = AiUtopiaSimEnv(
    {"active_roles": ["gatherer"], "backend": "sim", "max_episode_ticks": sc.max_ticks}
)
obs, _ = env.reset(seed=sc.seed)
state = {k: v.to(device) for k, v in mod.get_initial_state().items()}


def batch(v):
    if isinstance(v, dict):
        return {k: batch(x) for k, x in v.items()}
    return torch.as_tensor(np.asarray(v)).unsqueeze(0).to(device)


names = ["NAV", "HARV", "DEP", "SRCH", "WAIT", "NOOP"]
skills = []
for _step in range(20):
    ao = obs["gatherer_0"]
    si = {k: v.unsqueeze(0) for k, v in state.items()}
    with torch.no_grad():
        out = mod._forward_inference(
            {Columns.OBS: {k: batch(v) for k, v in ao.items()}, Columns.STATE_IN: si}
        )
    act = _greedy_decode(out[Columns.ACTION_DIST_INPUTS][0], ao.get("action_mask"))
    skills.append(names[int(act["skill_type"])])
    state = {k: v.squeeze(0) for k, v in out[Columns.STATE_OUT].items()}
    obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
    if term.get("gatherer_0") or trunc.get("gatherer_0"):
        break
oak = int(obs["gatherer_0"]["inv_slot_counts"][1:].sum())
env.close()
print(f"seed_1 greedy trajectory skills: {skills}")
print(f"NAVIGATE present: {'NAV' in skills} | final oak: {oak}")
