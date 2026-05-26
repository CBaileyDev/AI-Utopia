"""`aiutopia determinism check` — runs the seeded-replay harness.

M0: prints a friendly "not runnable yet" message because there are no
trained weights to load. Wired fully in M1 once the first gatherer
checkpoint exists."""
from __future__ import annotations

from pathlib import Path

import typer

from aiutopia.determinism.harness import configure_cuda_determinism


app = typer.Typer(no_args_is_help=True)


@app.command("check")
def check(
    weights: Path  = typer.Option(..., help="path to policy checkpoint"),
    episodes: int  = typer.Option(10, help="number of seed pairs to compare"),
) -> None:
    configure_cuda_determinism()
    if not weights.exists():
        typer.echo(f"weights not found: {weights}", err=True)
        raise typer.Exit(code=2)
    typer.echo("M0: determinism harness scaffold is in place but cannot run "
               "without real RLlib weights. See IMPLEMENTATION_PLAN.md task "
               "M1.X (gatherer first checkpoint).")
    raise typer.Exit(code=3)
