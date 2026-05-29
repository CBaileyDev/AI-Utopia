"""N20 — root-cause probe for NON-DETERMINISTIC consecutive HARVEST.

Finding under investigation: back-to-back HARVEST(oak_log, cap=1) on warm
instance-1 sometimes collects 1 log on EVERY dispatch (n14: 6/6) and
sometimes stalls after the 1st (golden-trace: 1, 0, 0). Agent position
reported CONSTANT in the stalling case.

This probe drives FabricBridge directly (max control over inter-dispatch
timing + raw obs), and per dispatch logs the discriminators the advisor
flagged:
  - full-precision pos BEFORE and AFTER (did the agent MOVE or was it pinned?)
  - velocity + on-ground proxy (y) before/after
  - resultCode + failureReason VERBATIM (the #1 path discriminator)
  - world-tick consumed (diff of obs 'tick_in_episode' = getOverworld().getTime(),
    which always increments; time_of_day is frozen by the gamerule). A clean
    in-place break burns ~1-3 ticks; a STALL burns STALL_TICK_BUDGET(20)+ ticks.
  - obs nearest_resource_distance BEFORE (mask/skill metric-mismatch check)
  - the chosen target is not directly exposed, but we infer the path from
    failureReason + tick burn + pos delta.

Conditions, each repeated 3x (fresh reset each repeat):
  A) reset -> immediate back-to-back HARVEST x8  (n14-like, agent still settling)
  B) reset -> WAIT(settle) -> HARVEST x8         (golden-trace-like, agent at rest)
  C) reset -> [HARVEST, small-NAVIGATE] interleaved x8 (does moving unstick it?)

INVESTIGATION ONLY. Does not modify production code. Uses warm instance-1.
"""
from __future__ import annotations

import json
import sys
import time
import uuid

from py4j.java_gateway import GatewayParameters, JavaGateway

PORT = 25001
PLAYER = "gatherer_0"
SEED = 1


def _p(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _obs_for(ep, player):
    raw = json.loads(str(ep.observationsAll()))
    return raw.get(player, {})


def _pos(o):
    p = o.get("position", [None, None, None])
    return (float(p[0]), float(p[1]), float(p[2])) if p and p[0] is not None else (None, None, None)


def _vel(o):
    v = o.get("velocity", [0.0, 0.0, 0.0])
    return (float(v[0]), float(v[1]), float(v[2]))


def _wtick(o):
    # tick_in_episode = getOverworld().getTime() % 24000 (always increments).
    t = o.get("tick_in_episode", [0])
    return int(t[0]) if isinstance(t, list) else int(t)


def _nrd(o):
    d = o.get("nearest_resource_distance", [999.0])
    return float(d[0]) if isinstance(d, list) else float(d)


def _n_logs_in_grid(o):
    # g_resource_grid is 32*32*6 flat; channel 0 = log. Count log cells.
    g = o.get("g_resource_grid", [])
    if not g:
        return -1
    n = 0
    for i in range(0, len(g), 6):
        if g[i] >= 0.5:
            n += 1
    return n


def _inv_logs(o):
    # Count items via inv_slot_counts (we only care about non-empty growth).
    ids = o.get("inv_slot_item_ids", [])
    cts = o.get("inv_slot_counts", [])
    total = 0
    for i, c in zip(ids, cts):
        if int(i) != 0:  # item id 0 = air/empty in ItemIdTable
            total += int(c)
    return total


def _dispatch_one(ep, motor, action_dict, label):
    """Dispatch a single skill, wait for its completion event, return matched evt."""
    inv = f"{label}-{uuid.uuid4().hex[:6]}"
    motor.dispatchSkill(PLAYER, json.dumps(action_dict), inv)
    evs = list(ep.advanceTickAwaitEvents(15_000))
    matched = None
    residual = []
    for e in evs:
        es = str(e)
        if inv in es:
            matched = json.loads(es)
        else:
            residual.append(es[:120])
    return matched, residual


def _harvest_action(cap_blocks=1, timeout_ticks=400):
    return {
        "skill_type": 1,
        "target_class": 0,  # oak_log
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [cap_blocks / 64.0],
        "timeout_ticks": timeout_ticks,
    }


def _wait_action(ticks=40):
    # WAIT = skill_type 4; duration from scalar_param * MAX_DURATION(~200).
    return {
        "skill_type": 4,
        "target_class": 0,
        "spatial_param": [0.0, 0.0, 0.0],
        "scalar_param": [ticks / 200.0],
        "timeout_ticks": 400,
    }


def _small_nav_action(dz=2.0):
    # NAVIGATE a couple blocks +z then back, to test whether MOVING unsticks.
    return {
        "skill_type": 0,
        "target_class": 0,
        "spatial_param": [0.0, 0.0, dz / 32.0],
        "scalar_param": [0.0],
        "timeout_ticks": 400,
    }


def _reset(ep):
    ep.resetWorld(SEED)
    ok = bool(ep.resetEpisode(PLAYER, SEED))
    return ok


def _run_harvest_sequence(ep, motor, n, pre=None, interleave_nav=False):
    """Run n HARVEST dispatches; optionally a `pre` action first; optionally
    interleave a small NAVIGATE between each HARVEST. Returns list of records."""
    if pre is not None:
        _dispatch_one(ep, motor, pre, "pre")
    records = []
    for i in range(n):
        o_before = _obs_for(ep, PLAYER)
        rec = {
            "i": i,
            "pos_before": _pos(o_before),
            "vel_before": _vel(o_before),
            "wtick_before": _wtick(o_before),
            "nrd_before": _nrd(o_before),
            "grid_logs_before": _n_logs_in_grid(o_before),
            "inv_before": _inv_logs(o_before),
        }
        matched, residual = _dispatch_one(ep, motor, _harvest_action(), f"h{i}")
        o_after = _obs_for(ep, PLAYER)
        rec["pos_after"] = _pos(o_after)
        rec["wtick_after"] = _wtick(o_after)
        rec["grid_logs_after"] = _n_logs_in_grid(o_after)
        rec["inv_after"] = _inv_logs(o_after)
        rec["rc"] = matched.get("resultCode", "NO_EVENT") if matched else "NO_EVENT"
        rec["fr"] = matched.get("failureReason", "") if matched else ""
        rec["residual"] = residual
        records.append(rec)
        if interleave_nav:
            _dispatch_one(ep, motor, _small_nav_action(2.0), f"nav{i}")
    return records


def _fmt_pos(p):
    if p[0] is None:
        return "(?, ?, ?)"
    return f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f})"


def _print_records(tag, records):
    _p(f"--- {tag} ---")
    for r in records:
        dpos = "MOVED" if (
            r["pos_before"][0] is not None and r["pos_after"][0] is not None and (
                abs(r["pos_before"][0] - r["pos_after"][0]) > 1e-4
                or abs(r["pos_before"][2] - r["pos_after"][2]) > 1e-4
            )
        ) else "still"
        wburn = r["wtick_after"] - r["wtick_before"]
        if wburn < 0:
            wburn += 24000
        invd = r["inv_after"] - r["inv_before"]
        gridd = r["grid_logs_after"] - r["grid_logs_before"]
        _p(
            f"  [{r['i']}] rc={r['rc']:<16} invd={invd:+d} gridLogsd={gridd:+d} "
            f"wtickBurn={wburn:>4} nrd={r['nrd_before']:6.2f} {dpos} "
            f"posB={_fmt_pos(r['pos_before'])} posA={_fmt_pos(r['pos_after'])}"
        )
        if r["fr"]:
            _p(f"        fr={r['fr']!r}")
        for res in r["residual"]:
            _p(f"        residual: {res}")


def main() -> int:
    gw = JavaGateway(gateway_parameters=GatewayParameters(
        port=PORT, auto_convert=True, auto_field=False))
    ep = gw.entry_point
    motor = ep.motorBridge()
    _p(f"[n20] health={ep.health()!r}")

    # Re-attach fake player (idempotent if already present).
    ep.carpetSpawn(PLAYER, "", "gatherer")
    time.sleep(0.3)

    REPEATS = 3
    N = 8

    for cond, (pre, nav) in {
        "A_immediate": (None, False),
        "B_settled":   (_wait_action(60), False),
        "C_nav_interleaved": (None, True),
    }.items():
        for rep in range(REPEATS):
            ok = _reset(ep)
            time.sleep(0.4)  # let arena settle + agent land
            o0 = _obs_for(ep, PLAYER)
            _p(
                f"\n==== COND={cond} rep={rep} reset_ok={ok} "
                f"pos0={_fmt_pos(_pos(o0))} nrd0={_nrd(o0):.2f} "
                f"gridLogs0={_n_logs_in_grid(o0)} ===="
            )
            records = _run_harvest_sequence(ep, motor, N, pre=pre, interleave_nav=nav)
            collected = sum(1 for r in records if r["rc"] == "COMPLETED" and (r["inv_after"] - r["inv_before"]) > 0)
            _print_records(f"{cond} rep={rep}", records)
            _p(f"  => COMPLETED-with-collection: {collected}/{N}")

    gw.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
