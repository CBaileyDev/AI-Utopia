"""Ray Tune entry point for AI Utopia training (M1)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# CUBLAS workspace must be set BEFORE CUDA init for deterministic LSTM:
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import ray
from ray import tune

# Use ray.tune.CheckpointConfig — ray.train.CheckpointConfig deprecates
# checkpoint_frequency/at_end, but Tune's internal trial machinery still
# expects them as ints, leading to "int % str" crashes mid-iter.
from ray.tune import CheckpointConfig, RunConfig

from aiutopia.common.config import Paths
from aiutopia.common.logging import get_logger, setup_logging
from aiutopia.train.callbacks import (
    AiUtopiaMetricsCallback,
    EvalGateStopCallback,
    ExploitHuntCallback,
    M1EvalScenarioCallback,
)
from aiutopia.train.config import m1_gatherer_config, multi_agent_config

log = get_logger("train")


def _make_callbacks_class(env_config: dict, eval_interval: int):
    """Compose multiple RLlibCallbacks into one. In Ray 2.40+ .callbacks(...)
    accepts a list too, but a single class is the most portable form."""
    try:
        from ray.rllib.callbacks.callbacks import RLlibCallback
    except ImportError:
        from ray.rllib.algorithms.callbacks import DefaultCallbacks as RLlibCallback

    metrics       = AiUtopiaMetricsCallback()
    exploit       = ExploitHuntCallback(every_n_iters=200)
    eval_scenario = M1EvalScenarioCallback(eval_interval=eval_interval,
                                             env_config=env_config)
    gate          = EvalGateStopCallback(milestone="M1")
    delegates = [metrics, exploit, eval_scenario, gate]

    class _Composite(RLlibCallback):
        def on_train_result(self, *, algorithm, metrics_logger=None,
                              result, **kwargs):
            for cb in delegates:
                cb.on_train_result(algorithm=algorithm,
                                    metrics_logger=metrics_logger,
                                    result=result, **kwargs)
    return _Composite


def main() -> None:
    setup_logging("INFO")
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", default="M1", choices=["M1"])
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--seed",       type=int, default=1)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--num-env-runners", type=int, default=4,
                        help="Ray EnvRunner workers (= # of Fabric instances)")
    parser.add_argument("--num-envs-per-runner", type=int, default=1,
                        help="envs per worker (1 keeps each env pinned to its own agent)")
    parser.add_argument("--num-learners", type=int, default=0,
                        help="Ray Learner processes. 0 runs Learner in driver "
                             "(required on Windows where PyTorch lacks libuv).")
    parser.add_argument("--roles", default="gather", type=str,
                        help="Comma-separated role list: 'gather' (default, M1B), "
                             "'gather,explore' (Explorer M2a), "
                             "'gather,explore,farm' (full M2). "
                             "Backward compat: 'gather' → m1_gatherer_config, "
                             "multi-role → multi_agent_config.")
    parser.add_argument("--backend", default="real", choices=["real", "sim"],
                        help="env backend: 'real' (live Minecraft via Py4J) or "
                             "'sim' (headless fast-sim AiUtopiaSimEnv).")
    parser.add_argument("--decision-core", action="store_true",
                        help="M2 (sim only): demote HARVEST to pointer-MINE + run the "
                             "2-cluster blind-explore arena + PBRS shaping, so the "
                             "POLICY learns instance-selection + explore-when-blind.")
    parser.add_argument("--natural-world", action="store_true",
                        help="(real backend only) Train on natural Minecraft world: "
                             "peaceful=True (no mobs), arena_bounds_check=False (lift truncation), "
                             "tick_warp=True (for speed).")
    parser.add_argument("--spawn-jitter", type=float, default=0.0,
                        help="(sim only) +/- blocks to jitter agent spawn each training "
                             "episode so some start HARVEST-masked, forcing NAVIGATE learning.")
    parser.add_argument("--approach-shaping", action="store_true",
                        help="(sim only) PBRS distance-reduction toward nearest log while "
                             "HARVEST masked, so jittered masked spawns are solvable.")
    args = parser.parse_args()

    paths = Paths.from_env(); paths.ensure()
    # num_cpus must exceed num_env_runners (each EnvRunner reserves ~1 CPU slot)
    # + the in-driver learner/driver. EnvRunners are I/O-bound (waiting on Py4J
    # while the JVMs compute), so reserving >physical-core slots is safe here.
    ray.init(num_cpus=max(16, args.num_env_runners + 6), num_gpus=1,
             object_store_memory=8 * 1024**3,
             _system_config={"object_spilling_threshold": 0.95})

    # Parse roles: "gather" → ["gatherer"], "gather,explore,farm" → ["gatherer", "explorer", "farmer"]
    role_abbrevs = [r.strip() for r in args.roles.split(",")]
    role_map = {"gather": "gatherer", "explore": "explorer", "farm": "farmer"}
    roles = [role_map.get(abbrev, abbrev) for abbrev in role_abbrevs]

    # Dispatch to appropriate config factory
    if roles == ["gatherer"]:
        # M1B backward compat: use single-role config
        cfg = m1_gatherer_config(
            backend=args.backend,
            seed=args.seed,
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            decision_core=args.decision_core,
            natural_world=args.natural_world,
            spawn_jitter=args.spawn_jitter,
            approach_shaping=args.approach_shaping,
        )
    else:
        # M2 multi-role MAPPO
        cfg = multi_agent_config(
            roles=roles,
            backend=args.backend,
            seed=args.seed,
            num_env_runners=args.num_env_runners,
            num_envs_per_env_runner=args.num_envs_per_runner,
            natural_world=args.natural_world,
        )
    # Override num_learners for Windows-libuv compat (T7.5 finding).
    if args.num_learners == 0:
        cfg = cfg.learners(num_learners=0, num_gpus_per_learner=1.0)
    env_config = cfg.env_config if hasattr(cfg, "env_config") else \
                  cfg.to_dict().get("env_config", {})
    cfg = cfg.callbacks(_make_callbacks_class(env_config, args.evaluation_interval))

    run_id = f"aiutopia_{args.milestone}_seed{args.seed}"
    tuner = tune.Tuner(
        "PPO",
        param_space=cfg,                       # PASS CONFIG OBJECT, not dict
        run_config=RunConfig(
            name=run_id,
            storage_path=str(paths.runs_dir),
            checkpoint_config=CheckpointConfig(
                num_to_keep=10,
                checkpoint_frequency=50,
                checkpoint_at_end=True,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
            ),
            # NOTE(T21 finding): Ray 2.55 doesn't expose custom_metrics/* as a
            # Tune stop-criterion key path. The gate is still emitted into the
            # result dict by EvalGateStopCallback for observability, but Tune
            # itself only stops on training_iteration here. Use the
            # m1b-evaluation-gate.sh script post-hoc to verify gate passage.
            stop={"training_iteration": args.max_iters},
            verbose=1,
        ),
    )
    log.info("starting training: %s", run_id)
    results = tuner.fit()
    best = results.get_best_result(metric="env_runners/episode_return_mean",
                                     mode="max")
    log.info("best checkpoint: %s", best.checkpoint)

    if best.checkpoint is not None:
        metrics_file = Path(best.checkpoint.path) / "aiutopia_metrics.json"
        final = best.metrics or {}
        sampler = final.get("env_runners", {}) or final.get("sampler_results", {})
        stats = sampler.get("episode_extra_stats", {})
        out = {
            "last_50k_steps_return_trend": float(
                sampler.get("episode_return_mean", 0)),
            "evaluation_scenario_success_rate":  float(
                stats.get("eval_m1_oak_log_success_rate", 0.0)),
            "exploit_penalty_ratio":       float(
                stats.get("exploit_total_per_episode", 0.0)),
            "per_role_entropy":            {"gatherer": float(
                final.get("custom_metrics", {}).get("gatherer_policy/entropy", 0))},
            "q_variance_ratio":            1.0,
            "trajectory_cosine":           1.0,
            "determinism_argmax_div":      1.0,   # T19 overwrites
            "determinism_l2":              999.0,
        }
        metrics_file.write_text(json.dumps(out, indent=2))
        log.info("wrote %s", metrics_file)


if __name__ == "__main__":
    main()
