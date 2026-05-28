"""N15c: Bisect which action field breaks HARVEST dispatch.

Reuses ONE Py4J connection. Between trials, waits the FULL skill timeout
(~14s @ 800 ticks) so we don't conflate pre-emption with completion.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from py4j.java_gateway import GatewayParameters, JavaGateway


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def trial(motor, ep, label: str, action: dict, wait_s: int = 15) -> None:
    inv = f"trial-{label}-{uuid.uuid4().hex[:6]}"
    enc = json.dumps(action)
    _p(f"[bis] {label}: encoded len={len(enc)}, fields={list(action.keys())}")
    motor.dispatchSkill("gatherer_0", enc, inv)
    t0 = time.time()
    evs = list(ep.advanceTickAwaitEvents(wait_s * 1000))
    dur = time.time() - t0
    _p(f"[bis] {label}: poll dur={dur:.2f}s, n_events={len(evs)}")
    matched = False
    for e in evs:
        e_str = str(e)
        if inv in e_str:
            _p(f"[bis] {label}: ★ matched event for {inv}")
            _p(f"[bis]   {e_str[:350]}")
            matched = True
        else:
            _p(f"[bis] {label}: residual event (other invocation):")
            _p(f"[bis]   {e_str[:200]}")
    if not matched:
        _p(f"[bis] {label}: ✗ NO MATCHED EVENT — dispatch lost or skill stuck")


def main() -> int:
    PORT = 25001
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()
    _p(f"[bis] health={ep.health()!r}")

    base = {
        "skill_type": 1, "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [1.0 / 64.0],
        "timeout_ticks": 600,    # 10s @ 60 TPS
    }
    # T0: minimal (known good from n15)
    trial(motor, ep, "T0_minimal", dict(base))

    # T1: + should_broadcast (single field add)
    trial(motor, ep, "T1_should_bcast", {**base, "should_broadcast": 0})

    # T2: + comm_target_mask
    trial(motor, ep, "T2_target_mask", {**base, "comm_target_mask": [0, 0, 0, 0]})

    # T3: + small comm_payload (8 elements)
    trial(motor, ep, "T3_pay8", {**base, "comm_payload": [0.0] * 8})

    # T4: + full comm_payload (128 elements)
    trial(motor, ep, "T4_pay128", {**base, "comm_payload": [0.0] * 128})

    # T5: ALL extra fields together
    trial(motor, ep, "T5_full", {**base,
        "should_broadcast": 0,
        "comm_target_mask": [0, 0, 0, 0],
        "comm_payload": [0.0] * 128,
    })

    return 0


if __name__ == "__main__":
    sys.exit(main())
