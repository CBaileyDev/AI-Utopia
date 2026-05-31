#!/usr/bin/env python
"""Phase 3: Persistent Survival World Validation.

Run Lumberjack on real Minecraft survival world (port 25100).
Measures: survival time, oak_log collection, hunger/health progression.

Run: PYTHONPATH=src PY4J_PRODUCTION_PORT=25100 py -3.11 scripts/phase3_persistent_survival.py
     --max-ticks 2000 --seed 42
"""
from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

OAK_LOG_ITEM_ID = 132  # from aiutopia.env.reward._ITEM_ID_TO_NAME


def _inv_count(obs: dict, item_id: int) -> int:
    """Sum inventory counts for slots holding item_id.

    Honest fix: prior code read obs['oak_log'] which is ABSENT from the obs dict —
    so it always returned the hardcoded default 0. Oak logs live in
    inv_slot_item_ids / inv_slot_counts (parallel arrays).
    """
    ids = obs.get("inv_slot_item_ids", []) or []
    counts = obs.get("inv_slot_counts", []) or []
    return int(sum(int(c) for i, c in zip(ids, counts) if int(i) == item_id))


def main() -> None:
    """Phase 3: persistent survival world test."""
    from aiutopia.common.logging import get_logger, setup_logging
    from aiutopia.env.bridge import FabricBridge
    from aiutopia.env.spaces import build_role_action_space

    setup_logging("INFO")
    log = get_logger("phase3")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ticks", type=int, default=2000, help="Max ticks (20 TPS ≈ 100 sec)")
    parser.add_argument("--seed", type=int, default=42, help="World seed")
    parser.add_argument("--port", type=int, default=25100, help="Py4J production port")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("PHASE 3: Persistent Survival World Validation")
    log.info("=" * 70)
    log.info(f"Max ticks: {args.max_ticks} ({args.max_ticks / 20:.1f} sec at 20 TPS)")
    log.info(f"World seed: {args.seed}")
    log.info(f"Py4J port: {args.port}")

    # Connect to production server
    try:
        bridge = FabricBridge(port=args.port)
        bridge.open()
        health = bridge.health()
        log.info(f"Server health: {health}")
    except Exception as e:
        log.error(f"Failed to connect to production server: {e}")
        return

    # Setup agent
    agent_id = "lumberjack_0"
    try:
        bridge.carpet_spawn(agent_id, skin="", role="gatherer")
        log.info(f"✓ Agent {agent_id} spawned")
    except Exception:
        log.info(f"Agent {agent_id} already exists (idempotent)")

    # Get action space (Gatherer)
    action_space = build_role_action_space("gatherer")

    # Episode metrics
    metrics = {
        "ticks": 0,
        "oak_log_final": 0,
        "health_final": 20.0,
        "hunger_final": 20.0,
        "time_of_day_final": 0,
        "position_final": [0, 0, 0],
        "deaths": 0,
        "avg_health": [],
        "avg_hunger": [],
    }

    log.info(f"Running {args.max_ticks} ticks with random Lumberjack policy...")
    log.info("")

    # Main loop
    for tick in range(args.max_ticks):
        try:
            # Random action
            action = {agent_id: action_space.sample()}

            # Dispatch + advance
            bridge.dispatch_skill(agent_id, action[agent_id], f"{agent_id}-{tick}")
            completion_jsons = bridge.advance_tick_await_events(timeout_ms=10_000)

            # Read obs
            obs_all = bridge.observations_all()
            obs = obs_all.get(agent_id, {})

            # Track metrics
            health = float(obs.get("health", [20.0])[0]) if obs.get("health") is not None else 20.0
            hunger = float(obs.get("hunger", [20.0])[0]) if obs.get("hunger") is not None else 20.0
            oak_log = _inv_count(obs, OAK_LOG_ITEM_ID)
            time_of_day = int(obs.get("time_of_day", [0])[0]) if obs.get("time_of_day") is not None else 0
            position = obs.get("position", [0, 0, 0])

            metrics["ticks"] = tick + 1
            metrics["oak_log_final"] = oak_log
            metrics["health_final"] = health
            metrics["hunger_final"] = hunger
            metrics["time_of_day_final"] = time_of_day
            metrics["position_final"] = position
            metrics["avg_health"].append(health)
            metrics["avg_hunger"].append(hunger)

            if health <= 0:
                metrics["deaths"] += 1
                log.warning(f"[{tick}] Agent died! (health={health})")
                break

            # Log progress every 250 ticks
            if (tick + 1) % 250 == 0:
                log.info(
                    f"[{tick + 1}] oak_log={oak_log}, health={health:.1f}, hunger={hunger:.1f}, "
                    f"time={time_of_day}, pos=({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
                )

        except Exception as e:
            log.error(f"Error at tick {tick}: {e}")
            break

    # Cleanup
    bridge.close()

    # Summary
    avg_health = sum(metrics["avg_health"]) / len(metrics["avg_health"]) if metrics["avg_health"] else 0
    avg_hunger = sum(metrics["avg_hunger"]) / len(metrics["avg_hunger"]) if metrics["avg_hunger"] else 0

    log.info("")
    log.info("=" * 70)
    log.info("PHASE 3 RESULTS")
    log.info("=" * 70)
    log.info(f"Ticks survived: {metrics['ticks']} / {args.max_ticks}")
    log.info(f"Oak logs collected: {metrics['oak_log_final']}")
    log.info(f"Deaths: {metrics['deaths']}")
    log.info(f"Final health: {metrics['health_final']:.1f} / 20")
    log.info(f"Final hunger: {metrics['hunger_final']:.1f} / 20")
    log.info(f"Avg health: {avg_health:.1f}")
    log.info(f"Avg hunger: {avg_hunger:.1f}")
    log.info(f"Final time_of_day: {metrics['time_of_day_final']} (0=dawn, 6000=day, 12000=dusk, 18000=night)")
    log.info(f"Final position: {metrics['position_final']}")

    # Success criteria
    log.info("")
    log.info("Success Criteria:")
    log.info(f"  ✓ Survived: {metrics['ticks'] > 0} (alive at end)")
    log.info(f"  {'✓' if metrics['oak_log_final'] > 0 else '✗'} Foraging: oak_log > 0 (collected: {metrics['oak_log_final']})")
    log.info(f"  {'✓' if metrics['deaths'] == 0 else '✗'} No deaths: deaths == 0 (actual: {metrics['deaths']})")
    log.info("")

    # Write results JSON
    results_file = Path("Research") / "PHASE_3_SURVIVAL_RESULTS.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
