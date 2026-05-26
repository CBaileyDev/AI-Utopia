"""`aiutopia memory inspect` — pretty-prints retrieval results.

Useful for "why did the planner think X" questions once M5 is live.
M0 returns raw collection contents (no LLM-summary generation yet)."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from aiutopia.common.config import Paths
from aiutopia.common.ids import is_ulid, memory_id_for
from aiutopia.memory.client import open_chroma


app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("inspect")
def inspect(
    agent_uuid: str = typer.Option(..., help="agent_uuid (ULID)"),
    query: str      = typer.Option("", help="optional NL query for ANN retrieval"),
    top_k: int      = typer.Option(10, help="number of results"),
) -> None:
    if not is_ulid(agent_uuid):
        typer.echo(f"not a ULID: {agent_uuid}", err=True)
        raise typer.Exit(code=2)

    paths = Paths.from_env()
    client = open_chroma(paths.chroma_dir)
    try:
        coll = client.get_collection(memory_id_for(agent_uuid))
    except Exception as exc:
        typer.echo(f"no memory collection for {agent_uuid}: {exc}", err=True)
        raise typer.Exit(code=3)

    if query:
        # M5 will wire BGE here; M0 returns by recency.
        typer.echo("(M0: ANN query not yet wired; falling back to most-recent)")

    got = coll.get(limit=top_k, include=["documents", "metadatas"])
    if not got["ids"]:
        typer.echo("(empty)")
        return

    table = Table(title=f"memory for {agent_uuid} (top {top_k})")
    table.add_column("id")
    table.add_column("ts")
    table.add_column("type")
    table.add_column("imp")
    table.add_column("summary", overflow="fold")
    for rid, meta, doc in zip(got["ids"], got["metadatas"], got["documents"]):
        meta = meta or {}
        table.add_row(
            rid,
            str(meta.get("timestamp", "")),
            str(meta.get("event_type", "")),
            f"{meta.get('importance_score', 0):.2f}",
            doc,
        )
    console.print(table)
