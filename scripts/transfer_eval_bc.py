#!/usr/bin/env python
"""Real-MC transfer eval for a bare state_dict checkpoint (e.g. weights/bc_gatherer.pt).

Builds the gatherer RLModule, loads the state_dict, runs the 3 M1 gate scenarios on
REAL Minecraft via run_scenario(real backend). The real WorldOps arena is byte-faithful
to the sim, so seed_1 is genuinely HARVEST-masked at spawn there too -> true test of
navigate-then-harvest transfer.
Run: PYTHONPATH=src py -3.11 scripts/transfer_eval_bc.py --weights weights/bc_gatherer.pt --port 25001
"""
from __future__ import annotations
import argparse, importlib, os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main() -> None:
    import torch
    from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
    sp = importlib.import_module("aiutopia." + "env.spaces")
    rw = importlib.import_module("aiutopia." + "env.reward")
    from aiutopia.rl_module.role_rl_module import AiUtopiaRoleRLModule
    sr = importlib.import_module("aiutopia.train.scenario_runner")

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", type=int, default=25001)
    args = ap.parse_args()

    spec = MultiRLModuleSpec(rl_module_specs={"gatherer_policy": RLModuleSpec(
        module_class=AiUtopiaRoleRLModule,
        observation_space=sp.build_role_observation_space("gatherer", stage=1),
        action_space=sp.build_role_action_space("gatherer"),
        model_config={"role": "gatherer", "max_seq_len": 32, "actor_hidden": [256],
                      "mask_comm": True, "core_encoder": {"core_hidden": [512, 256]},
                      "shared_backbone": {"lstm_hidden": 256}, "ctde_critic": {"critic_hidden": 256}})}).build()
    mod = spec["gatherer_policy"]
    sd = torch.load(args.weights, map_location="cpu")
    missing, unexpected = mod.load_state_dict(sd, strict=False)
    print(f"[load] {args.weights} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    mod.eval()

    env_config = {"active_roles": ["gatherer"], "stage": 1, "py4j_ports": [args.port], "enable_memory_writes": False}
    results = []
    for sc in sr.M1_SCENARIOS:
        r = sr.run_scenario(sc, env_config=env_config, rl_module=mod)
        inv = rw._inventory_from_obs(r["final_inventory"].get("gatherer_0", {}))
        oak = sum(c for k, c in inv.items() if k == "oak_log")
        print(f"REAL gate {sc.name}: success={r['success']} oak={oak}", flush=True)
        results.append(r)
    print(f"REAL-MC TRANSFER success_rate: {sr.aggregate_success_rate(results):.3f}", flush=True)


if __name__ == "__main__":
    main()
