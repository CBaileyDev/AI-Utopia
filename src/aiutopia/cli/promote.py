"""`aiutopia promote-weights promote ...` — section 5.10 promotion CLI."""
from __future__ import annotations

from pathlib import Path

import typer

from aiutopia.common.config  import Paths
from aiutopia.promotion.service import promote_weights


app = typer.Typer(no_args_is_help=True)


@app.command("promote")
def promote(
    role:            str  = typer.Option(..., help="gatherer|builder|farmer|defender"),
    checkpoint:      Path = typer.Option(..., help="path to Ray checkpoint dir"),
    notes:           str  = typer.Option("", help="audit-log note"),
    skip_checklist:  bool = typer.Option(False, "--skip-checklist",
                                          help="bypass section 5.10 gates (dangerous)"),
) -> None:
    """Promote a Ray checkpoint to the production weights directory."""
    if not checkpoint.exists():
        typer.echo(f"checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(code=2)
    paths = Paths.from_env(); paths.ensure()
    if not skip_checklist:
        from aiutopia.promotion.checklist import run_checklist
        report = run_checklist(role=role, checkpoint=checkpoint, paths=paths)
        if not report.passes:
            typer.echo("section 5.10 promotion checklist FAILED:", err=True)
            for issue in report.issues:
                typer.echo(f"  - {issue}", err=True)
            raise typer.Exit(code=3)
        typer.echo(f"section 5.10 checklist: PASS ({len(report.gates_passed)} gates)")
    result = promote_weights(role_id=role,
                              checkpoint_dir=checkpoint,
                              paths=paths,
                              notes=notes)
    typer.echo(f"promoted {role}: v{result['from_version']} -> v{result['to_version']}")
    typer.echo(f"  weights: {result['weights_path']}")
    typer.echo(f"  deployment_id: {result['deployment_id']}")
