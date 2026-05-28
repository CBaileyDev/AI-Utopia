"""N18: end-to-end test of scenario_runner with a freshly-built RLModule.

Goal: prove the N17 nested-dict batching fix works against the real
AiUtopiaRoleRLModule + AiUtopiaPettingZooEnv path. A random policy is
fine — we just need scenarios to complete (no numpy.object_ crash) and
produce a success/failure verdict.
"""
from __future__ import annotations

import sys
import time

from aiutopia.env.spaces import build_role_observation_space, build_role_action_space
from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule
from aiutopia.train.scenario_runner import M1_SCENARIOS, run_scenario, aggregate_success_rate


def _p(msg): print(msg, file=sys.stderr, flush=True)


def main() -> int:
    _p("[n18] building AiUtopiaRoleRLModule (random init) …")
    obs_space = build_role_observation_space("gatherer", stage=1)
    act_space = build_role_action_space("gatherer")
    module = AiUtopiaRoleRLModule(
        observation_space=obs_space,
        action_space=act_space,
        model_config={
            "role": "gatherer",
            "max_seq_len": 32,
            "actor_hidden": [256],
            "core_encoder":    {"core_hidden": [512, 256]},
            "shared_backbone": {"lstm_hidden": 256},
            "ctde_critic":     {"critic_hidden": 256},
        },
    )
    module.eval()
    _p("[n18] module built")

    env_config = {
        "stage": 1,
        "active_roles": ["gatherer"],
        "seed_strategy": "fixed_easy",
        "py4j_ports": [25001],
        "tick_warp": True,
        "per_worker_seed_offset": False,
        "enable_memory_writes": False,
        "aiutopia_root_per_worker": False,
        "skill_timeout_ticks": {0: 800, 1: 800, 2: 800},
    }

    results = []
    for scenario in M1_SCENARIOS:
        t0 = time.time()
        _p(f"[n18] running {scenario.name} (max_ticks={scenario.max_ticks}) …")
        try:
            result = run_scenario(scenario, env_config=env_config, rl_module=module, device="cpu")
            dur = time.time() - t0
            _p(f"[n18]   {scenario.name}: success={result['success']} dur={dur:.1f}s")
            results.append(result)
        except Exception as e:
            dur = time.time() - t0
            _p(f"[n18]   {scenario.name}: ✗ EXCEPTION after {dur:.1f}s: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 1

    success_rate = aggregate_success_rate(results)
    _p(f"[n18] === success_rate = {success_rate:.2%} ({sum(1 for r in results if r['success'])}/{len(results)}) ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
