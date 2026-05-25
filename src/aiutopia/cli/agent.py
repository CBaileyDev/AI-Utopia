"""`aiutopia agent <spawn|kill|list>` — minimal M0 surface.

`spawn` does THREE things:
  1. Insert a row in identity.db (so the agent has a persistent identity)
  2. Create the Chroma collections `mem_{uuid}` and `skill_lib_{uuid}`
     (so `aiutopia memory inspect <uuid>` returns "(empty)" instead of
     "no memory collection")
  3. Call Py4J to Carpet `/player <name> spawn` so the visible Carpet
     fake player appears in the connected MC client.

If --no-fabric is passed, step 3 is skipped (useful for unit tests + when
no Fabric server is running). Steps 1 and 2 always run."""
from __future__ import annotations

import time
from pathlib import Path

import typer

from aiutopia.common.config import Paths, Py4JConfig
from aiutopia.common.ids import memory_id_for, skill_library_id_for
from aiutopia.common.logging import get_logger
from aiutopia.env.bridge import FabricBridge
from aiutopia.identity.service import IdentityService, init_identity_db
from aiutopia.identity.skin_pool import deterministic_skin_for_uuid, pick_name
from aiutopia.memory.client import open_chroma


app = typer.Typer(no_args_is_help=True)
log = get_logger(__name__)


@app.command("spawn")
def spawn(
    role: str           = typer.Option(..., help="gatherer|builder|farmer|defender"),
    name: str | None    = typer.Option(None, help="explicit agent name (else pool pick)"),
    no_fabric: bool     = typer.Option(False, "--no-fabric",
                                        help="skip Py4J Carpet /player spawn"),
    py4j_port: int      = typer.Option(0,
                                        help="override Py4J port (else production port)"),
) -> None:
    paths = Paths.from_env(); paths.ensure()
    init_identity_db(paths.identity_db,
                     Path("src/aiutopia/identity/migrations"))
    svc = IdentityService(paths.identity_db)

    role_obj = svc.get_role(role)   # type: ignore[arg-type]
    living_names = {a.agent_name for a in svc.list_living_agents()}
    chosen_name  = name or pick_name(role_obj.default_skin_pool,
                                      used=living_names)

    agent = svc.spawn_agent(role, chosen_name, born_at=int(time.time()))
    skin = deterministic_skin_for_uuid(agent.agent_uuid,
                                        role_obj.default_skin_pool)

    typer.echo(f"identity: spawned {chosen_name} ({role}, uuid={agent.agent_uuid})")

    # Always create Chroma collections so memory inspect doesn't 404
    chroma = open_chroma(paths.chroma_dir)
    chroma.get_or_create_collection(memory_id_for(agent.agent_uuid))
    chroma.get_or_create_collection(skill_library_id_for(agent.agent_uuid))
    typer.echo(f"memory:   collections mem_{agent.agent_uuid} + "
               f"skill_lib_{agent.agent_uuid} ready")

    if no_fabric:
        typer.echo("--no-fabric set; skipping Carpet /player spawn")
        return

    # The Fabric Py4J call is wired in T30 (this is the M0 stub).
    typer.echo("(carpet spawn wiring lands in T30)")


@app.command("kill")
def kill(
    agent_uuid: str = typer.Argument(..., help="agent_uuid (ULID)"),
    cause: str      = typer.Option("manual_cli", help="cause_of_death string"),
) -> None:
    paths = Paths.from_env()
    svc = IdentityService(paths.identity_db)
    svc.record_death(agent_uuid, died_at=int(time.time()), cause_of_death=cause)
    typer.echo(f"agent {agent_uuid} marked dead (cause: {cause})")


@app.command("list")
def list_agents() -> None:
    paths = Paths.from_env()
    svc = IdentityService(paths.identity_db)
    for a in svc.list_living_agents():
        typer.echo(f"  {a.agent_name:16} {a.role_id:9} uuid={a.agent_uuid}")
