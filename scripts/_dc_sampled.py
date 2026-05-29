"""Sampled (non-greedy) decision-core eval: is NAVIGATE latent under argmax?
Samples skill_type/target_class from the policy dist instead of argmax, on the
clusters arena, and reports NAVIGATE usage + how far south (toward cluster B) the
agent reaches + oak collected.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np
import torch
from ray.rllib.core import Columns

from transfer_eval import GATHERER_MODULE_DIR


def _p(m):
    print(m, file=sys.stderr, flush=True)


def _load():
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule  # noqa: F401
    from ray.rllib.core.rl_module.rl_module import RLModule
    m = RLModule.from_checkpoint(GATHERER_MODULE_DIR)
    m.eval()
    return m


def main():
    from aiutopia.env.reward import _inventory_from_obs
    from aiutopia.sim.sim_env import AiUtopiaSimEnv

    m = _load()
    gen = torch.Generator().manual_seed(0)

    def batch(v):
        if isinstance(v, dict):
            return {k: batch(x) for k, x in v.items()}
        return torch.as_tensor(np.asarray(v)).unsqueeze(0)

    for seed in (1, 2, 3):
        env = AiUtopiaSimEnv({
            "active_roles": ["gatherer"], "decision_core": True,
            "arena_mode": "clusters", "arena_half": 34.0, "max_episode_ticks": 400,
        })
        obs, _ = env.reset(seed=seed)
        o = obs["gatherer_0"]
        st = {k: v.unsqueeze(0) for k, v in m.get_initial_state().items()}
        hist = Counter()
        min_z = 0.0
        for _ in range(400):
            b = {k: batch(v) for k, v in o.items()}
            with torch.no_grad():
                out = m._forward_inference({Columns.OBS: b, Columns.STATE_IN: st})
            flat = out[Columns.ACTION_DIST_INPUTS][0]
            st = {k: v for k, v in out[Columns.STATE_OUT].items()}
            sk = int(torch.distributions.Categorical(logits=flat[268:274]).sample(generator=gen) if False
                     else torch.multinomial(torch.softmax(flat[268:274], -1), 1, generator=gen).item())
            tc = int(torch.multinomial(torch.softmax(flat[280:344], -1), 1, generator=gen).item())
            sp = flat[274:277].detach().cpu().numpy()
            act = {"skill_type": sk, "target_class": tc,
                   "spatial_param": np.clip(sp, -1, 1).astype(np.float32),
                   "scalar_param": np.array([1.0], np.float32),
                   "comm_payload": np.zeros(128, np.float32), "should_broadcast": 0,
                   "comm_target_mask": np.zeros(4, np.int8)}
            hist["NAVIGATE" if sk == 0 else ("MINE" if sk == 1 else f"sk{sk}")] += 1
            obs, _r, term, trunc, _i = env.step({"gatherer_0": act})
            o = obs["gatherer_0"]
            min_z = min(min_z, float(o["position"][2]))
            if not env.agents or term.get("gatherer_0") or trunc.get("gatherer_0"):
                break
        oak = int(sum(c for n, c in _inventory_from_obs(o).items() if n == "oak_log"))
        _p(f"  seed={seed} SAMPLED oak={oak}/64 NAVIGATE={hist.get('NAVIGATE',0)} MINE={hist.get('MINE',0)} min_z={min_z:.0f} (B~-72)")


if __name__ == "__main__":
    main()
