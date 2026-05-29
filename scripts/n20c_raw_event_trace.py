"""N20c — dump the RAW Py4J completion-event stream + server-side inventory
to pin the env-path stall.

n20b proved: env path (reset -> HARVEST x N) collects exactly 1 log, with the
COMPLETED event appearing on step 1 (not step 0) and every later step showing
rc=? (empty completion). n20 raw-bridge collected 8/8. This probe replays the
SAME dispatch shape the env uses (dispatch -> advanceTickAwaitEvents ->
observationsAll) BUT prints:
  - the verbatim raw event-JSON list returned by advanceTickAwaitEvents each step
  - the agentId field of each event (to catch player-name vs agent-id mis-keying)
  - whether the event arrived on the SAME step as its dispatch or one late
  - the agent's REAL inventory read from obs (server truth) each step

It mirrors the env's exact call order: dispatch THEN await THEN obs, with a
unique invocation id per dispatch (agent-{n}), and injects timeout_ticks=800
exactly like the env's skill_timeout_ticks config.

INVESTIGATION ONLY — warm instance-1 (25001). No production edits.
"""
from __future__ import annotations

import json
import sys
import time

from py4j.java_gateway import GatewayParameters, JavaGateway

PORT = 25001
PLAYER = "gatherer_0"
SEED = 1


def _p(msg):
    print(msg, file=sys.stderr, flush=True)


def _obs(ep):
    return json.loads(str(ep.observationsAll())).get(PLAYER, {})


def _inv_count(o):
    ids = o.get("inv_slot_item_ids", [])
    cts = o.get("inv_slot_counts", [])
    return sum(int(c) for i, c in zip(ids, cts) if int(i) != 0)


def main():
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()
    _p(f"[n20c] health={ep.health()!r}")
    ep.carpetSpawn(PLAYER, "", "gatherer")
    time.sleep(0.3)

    # Drain any stale events from prior runs.
    stale = list(ep.advanceTickAwaitEvents(200))
    if stale:
        _p(f"[n20c] drained {len(stale)} stale events before start")

    ep.resetWorld(SEED)
    ep.resetEpisode(PLAYER, SEED)
    time.sleep(0.4)

    # Replay env's EXACT call order: for each step, dispatch (unique inv id) then
    # advanceTickAwaitEvents (env uses 30_000ms) then observationsAll.
    counter = 0
    for step in range(8):
        counter += 1
        inv_id = f"{PLAYER}-{counter}"   # SAME shape as env: agent-{skill_counter}
        action = json.dumps({
            "skill_type": 1, "target_class": 0,
            "spatial_param": [0.0, 0.0, 0.0],
            "scalar_param": [1.0 / 64.0],
            "timeout_ticks": 800,   # SAME as env skill_timeout_ticks[1]
        })
        t0 = time.time()
        motor.dispatchSkill(PLAYER, action, inv_id)
        evs = [str(e) for e in ep.advanceTickAwaitEvents(30_000)]
        dur = time.time() - t0
        o = _obs(ep)
        inv = _inv_count(o)
        # Parse + classify each event.
        parsed = []
        for e in evs:
            try:
                ej = json.loads(e)
                parsed.append((ej.get("agentId", "?"), ej.get("skillInvocationId", "?"),
                               ej.get("resultCode", "?"), ej.get("failureReason", ""),
                               ej.get("clippedAxesBitset", 0)))
            except Exception:
                parsed.append(("PARSE_FAIL", e[:80], "", "", 0))
        matched_this = any(p[1] == inv_id for p in parsed)
        _p(f"[step {step}] dispatched inv_id={inv_id} dur={dur:.1f}s n_events={len(evs)} "
           f"matched_this_step={matched_this} server_inv={inv}")
        for aid, sid, rc, fr, clip in parsed:
            late = "  <-- LATE (prev step's skill)" if sid != inv_id else ""
            _p(f"    evt agentId={aid!r} invId={sid!r} rc={rc} clip={clip} fr={fr!r}{late}")
        if not evs:
            _p(f"    (NO EVENTS — advanceTickAwaitEvents returned empty after {dur:.1f}s)")

    gw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
