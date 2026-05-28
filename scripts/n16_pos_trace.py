"""N16: Decisive HARVEST diagnostic — pos trace + raw obs + back-to-back dispatches.

Goal: discriminate H1 (movement broken), H2 (chunk load), H3 (reach math),
H4 (breakBlock silent fail), H5 (state corruption) in a single run.

Flow:
  1. health()
  2. reset_episode(gatherer_0, seed=1)
  3. /data get entity gatherer_0 Pos    -> pos_after_reset
  4. /forceload query
  5. RAW worldOps.observationsAll() -> dump nearest_resource_distance + g_nearest_resources[0]
  6. dispatchSkill HARVEST(target=0, scalar=1/64, timeout=600). Poll every ~50ms
     and run /data get entity gatherer_0 Pos. Log timestamped pos.
  7. on completion log result + /say HARVEST_1_DONE_<r>
  8. WITHOUT reset: raw worldOps.observationsAll() again
  9. dispatchSkill HARVEST again same params; same pos trace
 10. on completion log result + /say HARVEST_2_DONE_<r>

No restart of Java. No commits. Reads-only over Py4J + runCommand.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from py4j.java_gateway import GatewayParameters, JavaGateway


PORT = 25001
AGENT = "gatherer_0"


def p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def get_pos(ep) -> str:
    # runCommand returns success-code; the actual response goes to the log.
    # We rely on the /data get... command being logged with float coords.
    # As a fallback, peek at obs (agent.position field) too.
    try:
        ep.runCommand(f"/data get entity {AGENT} Pos")
    except Exception as e:
        return f"<runCommand failed: {e}>"
    return ""


def raw_obs(ep, agent: str = AGENT) -> dict:
    j = ep.observationsAll()
    try:
        full = json.loads(j) if isinstance(j, str) else {}
    except Exception as e:
        p(f"[obs] JSON parse failed: {e}; first 200 chars: {str(j)[:200]}")
        return {}
    return full.get(agent, {})


def short_obs(obs: dict) -> str:
    if not obs:
        return "<empty>"
    pos = obs.get("position", "<no position>")
    nrd = obs.get("nearest_resource_distance", "<no nrd>")
    nr0 = None
    nr = obs.get("g_nearest_resources")
    if isinstance(nr, list) and nr:
        nr0 = nr[0]
    return f"pos={pos} nrd={nrd} g_nearest_resources[0]={nr0}"


def trace_dispatch(motor, ep, label: str, action: dict, total_s: int = 18) -> None:
    inv = f"{label}-{uuid.uuid4().hex[:6]}"
    enc = json.dumps(action)
    p(f"\n=== {label}: dispatching skill_type={action.get('skill_type')} target={action.get('target_class')} inv={inv} ===")

    # Mark in log so we can correlate via latest.log
    try:
        ep.runCommand(f"/say {label}_BEGIN_{inv}")
    except Exception:
        pass

    motor.dispatchSkill(AGENT, enc, inv)

    # Poll with very short timeouts; between polls, ping /data get pos
    deadline = time.time() + total_s
    n_polls = 0
    matched_event = None
    while time.time() < deadline:
        # Ping pos every iteration; the /data get... fires asynchronously
        # but lands in latest.log with a millisecond timestamp.
        try:
            ep.runCommand(f"/data get entity {AGENT} Pos")
        except Exception as e:
            p(f"[trace] {label} pos query failed: {e}")
        # Drain any completion events with a 100ms wait
        evs = list(ep.advanceTickAwaitEvents(100))
        n_polls += 1
        if evs:
            for e in evs:
                e_str = str(e)
                p(f"[trace] {label} event: {e_str[:300]}")
                if inv in e_str:
                    matched_event = e_str
            if matched_event:
                break
        time.sleep(0.05)

    p(f"[trace] {label}: polls={n_polls} matched={bool(matched_event)}")
    if matched_event is None:
        p(f"[trace] {label}: ✗ no terminal event for {inv} within {total_s}s")
    try:
        ep.runCommand(f"/say {label}_END_{inv}")
    except Exception:
        pass


def main() -> int:
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    p(f"[n16] health={ep.health()!r}")

    # 1. Reset episode
    ep.resetEpisode(AGENT, 1)
    # Tiny await so the spawn/clear/setblock race is gone
    time.sleep(0.5)
    ep.runCommand(f"/say N16_AFTER_RESET")
    ep.runCommand(f"/data get entity {AGENT} Pos")
    ep.runCommand(f"/forceload query")
    ep.runCommand(f"/data get entity {AGENT} abilities")
    ep.runCommand(f"/data get entity {AGENT} SelectedItem")

    # 2. Raw obs
    obs0 = raw_obs(ep)
    p(f"[n16] raw_obs after reset: {short_obs(obs0)}")
    # Also dump full g_nearest_resources (top-3)
    nr = obs0.get("g_nearest_resources", [])
    for i, row in enumerate(nr[:3]):
        p(f"[n16] g_nearest_resources[{i}] = {row}")

    base = {
        "skill_type": 1, "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [1.0 / 64.0],
        "timeout_ticks": 600,    # 10s @ 60 TPS
    }

    # 3. First HARVEST
    trace_dispatch(motor=ep.motorBridge(), ep=ep, label="HARVEST_1", action=dict(base))
    time.sleep(0.3)
    ep.runCommand(f"/say N16_AFTER_HARVEST_1")
    ep.runCommand(f"/data get entity {AGENT} Pos")

    # 4. Raw obs without reset
    obs1 = raw_obs(ep)
    p(f"[n16] raw_obs after HARVEST_1 (no reset): {short_obs(obs1)}")
    nr = obs1.get("g_nearest_resources", [])
    for i, row in enumerate(nr[:3]):
        p(f"[n16] g_nearest_resources[{i}] = {row}")

    # 5. Second HARVEST (no reset)
    trace_dispatch(motor=ep.motorBridge(), ep=ep, label="HARVEST_2", action=dict(base))
    time.sleep(0.3)
    ep.runCommand(f"/say N16_AFTER_HARVEST_2")
    ep.runCommand(f"/data get entity {AGENT} Pos")

    obs2 = raw_obs(ep)
    p(f"[n16] raw_obs after HARVEST_2: {short_obs(obs2)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
