"""N15b: Does the FULL probe-style action dict break dispatch?

n15 used a minimal 5-field action dict and got COMPLETED in 0.43s.
The probe uses an 8-field action dict (extra: should_broadcast,
comm_target_mask, comm_payload-128). Test whether the extra fields
or the 128-element payload break Java parsing.

Drives the same low-level Py4J path so wrapper isn't in the picture.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

import numpy as np
from py4j.java_gateway import GatewayParameters, JavaGateway


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> int:
    PORT = 25001
    PLAYER = "gatherer_0"
    TOKEN = uuid.uuid4().hex[:8]

    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()
    _p(f"[diag] health={ep.health()!r}")

    # Build the probe's exact action shape (post _to_python encoding)
    full_act = {
        "skill_type":       1,
        "target_class":     0,
        "spatial_param":    [0.0, 0.0, 0.0],
        "scalar_param":     [1.0 / 64.0],
        "should_broadcast": 0,
        "comm_target_mask": [0, 0, 0, 0],
        "comm_payload":     [0.0] * 128,
        "timeout_ticks":    800,
    }
    encoded = json.dumps(full_act)
    _p(f"[diag] encoded len={len(encoded)} preview={encoded[:200]}…")

    inv_id = f"diag-full-{TOKEN}"
    motor.dispatchSkill(PLAYER, encoded, inv_id)
    t0 = time.time()
    evs = list(ep.advanceTickAwaitEvents(8000))
    _p(f"[diag] full-fat dispatch → after {time.time()-t0:.2f}s, n={len(evs)} events")
    for e in evs:
        _p(f"[diag]   {str(e)[:400]}")

    # Now try with auto_convert=True (default) AND auto_field=True (the wrapper's setting)
    _p("[diag] reconnecting with auto_field=True (matches FabricBridge.open)")
    gw2 = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=True))
    ep2 = gw2.entry_point
    motor2 = ep2.motorBridge()
    inv_id2 = f"diag-autofield-{TOKEN}"
    motor2.dispatchSkill(PLAYER, encoded, inv_id2)
    t1 = time.time()
    evs2 = list(ep2.advanceTickAwaitEvents(8000))
    _p(f"[diag] auto_field=True dispatch → after {time.time()-t1:.2f}s, n={len(evs2)} events")
    for e in evs2:
        _p(f"[diag]   {str(e)[:400]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
