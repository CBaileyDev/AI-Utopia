"""N15: Minimal Py4J diagnostic to isolate where skill dispatch fails.

Tests 4 things in order, each with its own timeout:
  1) health() — must return "ok"
  2) runCommand("/say <token>") — Carpet command path
  3) motorBridge().dispatchSkill(<NONEXISTENT_PLAYER>, ...) — expect IMMEDIATE_FAILURE
     completion within 2s. If yes → dispatch path alive. If no → server.execute() broken.
  4) motorBridge().dispatchSkill(<REAL_PLAYER>, HARVEST, oak_log, timeout=200) —
     wait 5s. resultCode tells us what's going wrong with REAL dispatches.

No PettingZoo wrapper. No reward shaping. Just Py4J.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from py4j.java_gateway import GatewayParameters, JavaGateway


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    PORT = 25001
    PLAYER = "gatherer_0"
    BAD_PLAYER = "definitely_not_a_player_xyz"
    TOKEN = uuid.uuid4().hex[:8]

    _p(f"[diag] connecting Py4J port={PORT}")
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point

    # 1) health
    h = str(ep.health())
    _p(f"[diag] 1) health() → {h!r}")
    if h != "ok":
        _p("[diag] FAIL: server not attached")
        return 1

    # 2) runCommand
    cmd = f"/say PROBE_HELLO_{TOKEN}"
    ok = bool(ep.runCommand(cmd))
    _p(f"[diag] 2) runCommand({cmd!r}) → {ok}")

    # 3) Dispatch to nonexistent player — expect IMMEDIATE_FAILURE completion
    motor = ep.motorBridge()
    _p(f"[diag] 3a) motorBridge() returned: {motor!r}")
    bad_action = json.dumps({
        "skill_type": 1, "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [1.0/64.0],
        "timeout_ticks": 200,
    })
    bad_invocation = f"diag-bad-{TOKEN}"
    motor.dispatchSkill(BAD_PLAYER, bad_action, bad_invocation)
    _p(f"[diag] 3b) dispatchSkill({BAD_PLAYER}, …, {bad_invocation}) sent")
    events_3 = list(ep.advanceTickAwaitEvents(2000))
    _p(f"[diag] 3c) advanceTickAwaitEvents(2000) → n={len(events_3)} events")
    for ev in events_3:
        _p(f"[diag]    event: {str(ev)[:200]}")

    if len(events_3) == 0:
        _p("[diag] ★ ROOT CAUSE: dispatchSkill on a NONEXISTENT player produced NO completion event.")
        _p("[diag]   This means server.execute() runnables are not running, OR")
        _p("[diag]   the MotorBridge instance used by dispatchSkill differs from the one whose")
        _p("[diag]   completedQueue is polled by advanceTickAwaitEvents (singleton mismatch).")
        return 2

    # 4) Dispatch HARVEST to REAL player with short timeout
    good_action = json.dumps({
        "skill_type": 1, "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [1.0/64.0],
        "timeout_ticks": 300,    # 5s @ 60 TPS
    })
    good_invocation = f"diag-good-{TOKEN}"
    motor.dispatchSkill(PLAYER, good_action, good_invocation)
    _p(f"[diag] 4) dispatchSkill({PLAYER}, HARVEST oak_log, 300 ticks, {good_invocation})")
    t_start = time.time()
    events_4 = list(ep.advanceTickAwaitEvents(8000))
    _p(f"[diag]    advanceTickAwaitEvents(8000) returned after {time.time()-t_start:.2f}s "
       f"with n={len(events_4)} events")
    for ev in events_4:
        _p(f"[diag]    event: {str(ev)[:500]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
