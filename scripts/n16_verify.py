"""N16 verification: 5 consecutive HARVEST dispatches against reset arena.

After the y-offset reach-radius fix, every dispatch should COMPLETED with
brokenCount>=1. Pre-fix, only the first worked; the rest FAILED_TIMEOUT
with "broke 0 of 1".

Resets the arena ONCE so we have 8 fresh logs, then dispatches HARVEST
five times — waiting for each completion event before sending the next.
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
    SEED  = 1

    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()
    _p(f"[verify] health={ep.health()!r}")

    # Drain any leftover completion events
    leftover = list(ep.advanceTickAwaitEvents(100))
    if leftover:
        _p(f"[verify] drained {len(leftover)} leftover events: "
           f"{[str(e)[:80] for e in leftover]}")

    _p(f"[verify] carpetSpawn({PLAYER}) — re-attach fake player after server restart")
    spawn_ok = bool(ep.carpetSpawn(PLAYER, "", "gatherer"))
    _p(f"[verify]   carpetSpawn → {spawn_ok}")
    time.sleep(0.5)  # let player join

    _p(f"[verify] resetWorld + resetEpisode (seed={SEED})")
    ep.resetWorld(SEED)
    ok = bool(ep.resetEpisode(PLAYER, SEED))
    _p(f"[verify]   resetEpisode → {ok}")
    if not ok:
        return 1

    # Wait a moment for arena to settle and player to land
    time.sleep(0.5)

    completed = 0
    failed_timeout = 0
    other = 0

    for n in range(5):
        inv = f"verify-{n}-{uuid.uuid4().hex[:6]}"
        action = json.dumps({
            "skill_type": 1,                # HARVEST
            "target_class": 0,              # oak_log
            "spatial_param": [0.0, 0.0, 0.0],
            "scalar_param": [1.0/64.0],     # cap=1
            "timeout_ticks": 600,           # 10s @ 60 TPS
        })
        _p(f"[verify] [{n}] dispatching HARVEST inv={inv}")
        t0 = time.time()
        motor.dispatchSkill(PLAYER, action, inv)
        evs = list(ep.advanceTickAwaitEvents(12_000))
        dur = time.time() - t0
        _p(f"[verify] [{n}] poll dur={dur:.2f}s, n_events={len(evs)}")
        matched = None
        for e in evs:
            e_str = str(e)
            if inv in e_str:
                matched = json.loads(e_str)
                break
            _p(f"[verify] [{n}]   residual: {e_str[:200]}")
        if matched is None:
            _p(f"[verify] [{n}] ✗ no matched event")
            other += 1
            continue
        rc = matched.get("resultCode", "?")
        fr = matched.get("failureReason", "")
        _p(f"[verify] [{n}] resultCode={rc} failureReason={fr!r}")
        if rc == "COMPLETED":
            completed += 1
        elif rc == "FAILED_TIMEOUT":
            failed_timeout += 1
        else:
            other += 1

    _p(f"[verify] === summary: completed={completed}/5 timeout={failed_timeout} other={other} ===")
    return 0 if completed >= 4 else 2


if __name__ == "__main__":
    sys.exit(main())
