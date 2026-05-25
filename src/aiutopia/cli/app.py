"""§7.5 — Top-level Typer CLI surface."""
from __future__ import annotations

import typer

from aiutopia.cli import agent, memory, determinism
from aiutopia.common.logging import setup_logging


app = typer.Typer(
    name="aiutopia",
    help="AI Utopia — multi-agent Minecraft AI village.",
    add_completion=False,
    no_args_is_help=True,
)
app.add_typer(agent.app, name="agent",
              help="Spawn / kill / list AI agents.")
app.add_typer(memory.app, name="memory",
              help="Inspect agent episodic memory.")
app.add_typer(determinism.app, name="determinism",
              help="Determinism harness commands.")


@app.callback()
def _root_setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


if __name__ == "__main__":
    app()
