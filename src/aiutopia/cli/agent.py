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

    port = py4j_port or Py4JConfig.from_env().production_port
    log.info("connecting to Fabric Py4J on port %d", port)
    with FabricBridge(port=port) as bridge:
        if bridge.health() != "ok":
            typer.echo("ERROR: Fabric server reports unhealthy; aborting Carpet spawn",
                       err=True)
            raise typer.Exit(code=1)
        ok = bridge.carpet_spawn(chosen_name, skin=skin)
        if not ok:
            typer.echo(f"ERROR: Carpet /player {chosen_name} spawn failed", err=True)
            raise typer.Exit(code=2)
        # Plain ASCII arrow — Windows cp1252 console can't encode U+2192.
        typer.echo(f"carpet: /player {chosen_name} spawn (skin={skin}) -> ok")


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


@app.command("drive")
def drive(
    agent_name:  str       = typer.Option(..., help="agent name to drive"),
    skill:       int       = typer.Option(..., help="skill_type index (0=NAVIGATE,1=HARVEST,2=DEPOSIT_CHEST,3=SEARCH,4=WAIT,5=NOOP_BROADCAST)"),
    target:      int       = typer.Option(0,   help="target_class index"),
    dx:          float     = typer.Option(0.0, help="spatial param x in [-1,1]"),
    dy:          float     = typer.Option(0.0),
    dz:          float     = typer.Option(0.0),
    scalar:      float     = typer.Option(0.5, help="scalar_param in [0,1]"),
    py4j_port:   int       = typer.Option(25099, help="Py4J port"),
    timeout_ms:  int       = typer.Option(60_000, help="wait this long for completion"),
) -> None:
    """Manually dispatch a single skill to an agent and wait for completion.

    M1-Pipeline manual smoke tool — used to verify motor + obs + reward path
    without a trained RL policy. Plan B's training driver replaces this."""
    from aiutopia.env.bridge import FabricBridge
    import time, json as _json
    action = {
        "skill_type":       skill,
        "target_class":     target,
        "spatial_param":    [dx, dy, dz],
        "scalar_param":     [scalar],
        "comm_payload":     [0.0] * 128,
        "should_broadcast": 0,
        "comm_target_mask": [0, 0, 0, 0],
    }
    invocation_id = f"manual-{int(time.time()*1000)}"
    with FabricBridge(port=py4j_port) as bridge:
        bridge.dispatch_skill(agent_name, action, invocation_id)
        typer.echo(f"dispatched skill={skill} target={target} → {invocation_id}")
        # Block on completion
        events = bridge.advance_tick_await_events(timeout_ms=timeout_ms)
        if not events:
            typer.echo("timeout — no completion event arrived", err=True)
            raise typer.Exit(code=1)
        for evt in events:
            typer.echo(_json.dumps(evt, indent=2))
