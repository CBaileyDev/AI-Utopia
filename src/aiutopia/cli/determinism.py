"""`aiutopia determinism check --weights <ckpt>` — real-weights replay."""
from __future__ import annotations

import json
import os
# Critical: set BEFORE CUDA init (required for deterministic GPU LSTM):
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path

import typer

from aiutopia.common.config import Paths
from aiutopia.determinism.harness import (
    compute_divergence, configure_cuda_determinism, replay_with_rlmodule,
)


app = typer.Typer(no_args_is_help=True)


@app.command("check")
def check(
    weights:    Path = typer.Option(..., help="Ray algorithm checkpoint dir"),
    episodes:   int  = typer.Option(3, help="number of seed pairs"),
    py4j_port:  int  = typer.Option(25099, help="env Py4J port"),
) -> None:
    """Section 7.8 / 5.10 Gate 5 determinism check on real weights."""
    if not weights.exists():
        typer.echo(f"weights not found: {weights}", err=True)
        raise typer.Exit(code=2)
    configure_cuda_determinism()
    paths = Paths.from_env(); paths.ensure()

    # A Tune algorithm checkpoint isn't a MultiRLModule checkpoint —
    # use Algorithm.from_checkpoint to handle path resolution correctly.
    from ray.rllib.algorithms.algorithm import Algorithm
    typer.echo(f"loading checkpoint: {weights}")
    algo = Algorithm.from_checkpoint(str(weights))
    rl_module = algo.get_module("gatherer_policy")

    env_config = {
        "stage":                 1,
        "active_roles":          ["gatherer"],
        "seed_strategy":         "fixed_easy",
        "py4j_ports":            [py4j_port],
        "tick_warp":             True,
        "max_episode_ticks":     1000,
        "per_worker_seed_offset": False,
        "enable_memory_writes":  False,
    }

    divergences = []
    for ep in range(episodes):
        seed = ep + 1
        typer.echo(f"=== seed {seed}: running 2 replays ===")
        a = replay_with_rlmodule(rl_module, env_config=env_config,
                                   seed=seed, n_steps=1000)
        b = replay_with_rlmodule(rl_module, env_config=env_config,
                                   seed=seed, n_steps=1000)
        div = compute_divergence(a, b)
        divergences.append(div)
        verdict = "PASS" if div.passes else "FAIL"
        typer.echo(f"  argmax_div={div.action_argmax_divergence:.4f}  "
                    f"l2={div.continuous_param_l2:.4f}  {verdict}")

    all_pass = all(d.passes for d in divergences)
    avg_argmax = sum(d.action_argmax_divergence for d in divergences) / len(divergences)
    avg_l2     = sum(d.continuous_param_l2     for d in divergences) / len(divergences)

    metrics_file = weights / "aiutopia_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    else:
        metrics = {}
    metrics["determinism_argmax_div"] = avg_argmax
    metrics["determinism_l2"]         = avg_l2
    metrics_file.write_text(json.dumps(metrics, indent=2))
    typer.echo(f"\nUpdated {metrics_file}")
    typer.echo(f"  determinism_argmax_div: {avg_argmax:.4f}")
    typer.echo(f"  determinism_l2:         {avg_l2:.4f}")
    typer.echo(f"  overall: {'PASS' if all_pass else 'FAIL'}")
    algo.stop()
    raise typer.Exit(code=0 if all_pass else 3)
