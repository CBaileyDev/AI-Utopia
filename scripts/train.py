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
from ray.train import CheckpointConfig, RunConfig

from aiutopia.common.config import Paths
from aiutopia.common.logging import setup_logging, get_logger
from aiutopia.train.callbacks  import (
    AiUtopiaMetricsCallback,
    EvalGateStopCallback,
    ExploitHuntCallback,
    M1EvalScenarioCallback,
)
from aiutopia.train.config     import m1_gatherer_config


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
    args = parser.parse_args()

    paths = Paths.from_env(); paths.ensure()
    ray.init(num_cpus=16, num_gpus=1,
             object_store_memory=8 * 1024**3,
             _system_config={"object_spilling_threshold": 0.95})

    cfg = m1_gatherer_config(seed=args.seed)
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
                checkpoint_frequency=50,
                num_to_keep=10,
                checkpoint_at_end=True,
                checkpoint_score_attribute="env_runners/episode_return_mean",
                checkpoint_score_order="max",
            ),
            stop={
                "training_iteration":                       args.max_iters,
                "custom_metrics/M1/gate_passed":            0.5,  # >= 0.5 = passed
            },
            verbose=1,
            log_to_file=True,
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
