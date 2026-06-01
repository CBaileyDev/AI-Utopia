"""Deliverable-1 proof: roll the NAVIGATE-then-HARVEST demonstrator greedily and
report (a) its success rate on forced-masked spawns and (b) per-seed success on
the 3 fixed gate seeds incl seed_1, with NAVIGATE actually emitted on seed_1.

Note: VecGathererSim.step auto-resets a done env and returns the FRESH obs, so we
read success from the `terminated` flag (sim sets terminated = oak>=64) on the
terminating step, NOT from the post-step (reset) inventory.

Run:
  PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 scripts/bc_demonstrator_proof.py
"""

from __future__ import annotations

import numpy as np

from aiutopia.sim.bc_demonstrator import demonstrate
from aiutopia.sim.skills import SKILL_HARVEST, SKILL_NAVIGATE
from aiutopia.sim.vec_sim import VecGathererSim


def masked_success_rate(B: int = 256, max_steps: int = 60) -> float:
    sim = VecGathererSim(
        num_envs=B, max_episode_ticks=300, force_masked_spawn=True, randomize_layout=True
    )
    obs = sim.reset(np.arange(1, B + 1, dtype=np.int64))
    finished = 0
    succeeded = 0
    for _ in range(max_steps):
        obs, _rew, term, trunc = sim.step(demonstrate(obs))
        done = np.asarray(term, dtype=bool) | np.asarray(trunc, dtype=bool)
        finished += int(done.sum())
        succeeded += int(np.asarray(term, dtype=bool).sum())
    return succeeded / max(1, finished)


def gate_seed_proof(seed: int, max_steps: int = 1000) -> dict:
    sim = VecGathererSim(num_envs=1, max_episode_ticks=max_steps, randomize_layout=False)
    obs = sim.reset(np.array([seed], dtype=np.int64))
    masked_at_spawn = bool(obs["action_mask"]["skill_type"][0, SKILL_HARVEST] == 0)
    skills: list[int] = []
    success = False
    oak_at_term = 0
    for _ in range(max_steps):
        act = demonstrate(obs)
        skills.append(int(np.asarray(act["skill_type"]).reshape(-1)[0]))
        oak_pre = int(sim.oak[0])  # oak BEFORE the in-place skill mutation
        obs, _rew, term, trunc = sim.step(act)
        terminated = bool(np.asarray(term).reshape(-1)[0])
        if terminated:
            success = True
            oak_at_term = max(oak_pre, 64)  # terminated => oak>=64
            break
        if bool(np.asarray(trunc).reshape(-1)[0]):
            break
    return {
        "seed": seed,
        "masked_at_spawn": masked_at_spawn,
        "success": success,
        "oak": oak_at_term if success else int(sim.oak[0]),
        "navigate_emitted": SKILL_NAVIGATE in skills,
        "n_navigate": skills.count(SKILL_NAVIGATE),
        "n_harvest": skills.count(SKILL_HARVEST),
        "skill_prefix": skills[:8],
    }


def main() -> None:
    sr = masked_success_rate()
    print(f"oracle masked-spawn success rate: {sr:.3f}", flush=True)
    print("per-seed gate proof (clean vanilla-path seeds 1/2/3):", flush=True)
    all_ok = True
    for seed in (1, 2, 3):
        r = gate_seed_proof(seed)
        ok = r["success"]
        all_ok &= ok
        print(
            f"  seed {r['seed']}: success={r['success']} oak~{r['oak']:3d} "
            f"masked_at_spawn={r['masked_at_spawn']} "
            f"navigate_emitted={r['navigate_emitted']} "
            f"(n_nav={r['n_navigate']} n_harv={r['n_harvest']}) "
            f"skills[:8]={r['skill_prefix']} {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    print(f"DEMONSTRATOR GATE: {'3/3 PASS' if all_ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
