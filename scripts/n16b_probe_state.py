"""N16b: inspect agent state after first HARVEST stalls on (61, 66, -51).

Dispatches one HARVEST (which we expect to succeed), then asks the server
for agent position and the block state at (61, 66, -51) and immediate
neighbors.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from py4j.java_gateway import GatewayParameters, JavaGateway


def _p(msg): print(msg, file=sys.stderr, flush=True)


def main() -> int:
    PORT = 25001
    PLAYER = "gatherer_0"

    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()

    # Drain prior events
    list(ep.advanceTickAwaitEvents(100))

    # Spawn + reset
    ep.carpetSpawn(PLAYER, "", "gatherer")
    time.sleep(0.3)
    ep.resetWorld(1)
    ep.resetEpisode(PLAYER, 1)
    time.sleep(0.5)

    # Query initial pos
    _p("[s] initial pos:")
    ep.runCommand("/data get entity gatherer_0 Pos")
    time.sleep(0.3)

    # Dispatch first HARVEST
    inv = f"state-{uuid.uuid4().hex[:6]}"
    action = json.dumps({
        "skill_type": 1, "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [1.0/64.0],
        "timeout_ticks": 600,
    })
    motor.dispatchSkill(PLAYER, action, inv)
    evs = list(ep.advanceTickAwaitEvents(15_000))
    _p(f"[s] HARVEST 1: events={[str(e)[:200] for e in evs]}")

    # Where is agent now?
    _p("[s] post-HARVEST pos:")
    ep.runCommand("/data get entity gatherer_0 Pos")
    time.sleep(0.3)

    # Check the 8 ring positions
    _p("[s] checking ring blocks (post HARVEST 1):")
    ring = [
        (69, 66, -48), (68, 66, -45), (64, 66, -43), (60, 66, -45),
        (60, 66, -49), (61, 66, -51), (65, 66, -54), (69, 66, -51),
    ]
    for x, y, z in ring:
        ep.runCommand(f"/setblock {x} {y} {z} ~ keep")
        # ~ keep doesn't help; use data get pseudo
        ep.runCommand(f"/execute if block {x} {y} {z} oak_log run say BLOCK_OAK_AT_{x}_{y}_{z}")
        ep.runCommand(f"/execute if block {x} {y} {z} air run say BLOCK_AIR_AT_{x}_{y}_{z}")
        time.sleep(0.05)

    # 2nd HARVEST attempt
    _p("[s] HARVEST 2 (expected to find another log):")
    inv2 = f"state-{uuid.uuid4().hex[:6]}"
    motor.dispatchSkill(PLAYER, action, inv2)
    evs2 = list(ep.advanceTickAwaitEvents(15_000))
    _p(f"[s] HARVEST 2 events: {[str(e)[:300] for e in evs2]}")
    ep.runCommand("/data get entity gatherer_0 Pos")
    time.sleep(0.3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
