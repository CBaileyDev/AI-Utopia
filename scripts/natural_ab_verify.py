"""Focused A/B perception verifier for the natural-terrain obs fix.

Runs ONLY section A of scripts/natural_recon.py (perception_ab) — the clean
log-under-leaf A/B verdict — and prints the numbers as JSON. Does NOT overwrite
Research/NATURAL_RECON.md and does NOT run the HARVEST / natural-forest sections.
Reuses natural_recon.perception_ab verbatim so the harness can never drift from
the recon that established the bug.

PASS criterion (after the GathererOverlayBuilder topmost-LOG fix):
  - control bare-log column SEEN   (unchanged: grid_log_cells 0->1, dist ~3.6)
  - hypothesis leafed column SEEN  (THE FIX: grid_log_cells 0->1, dist ~correct)
  -> hypothesis_confirmed == False  (leaves NO LONGER occlude the log)

Before the fix this prints hypothesis_confirmed == True (leafed invisible).
Leaves instance-1 warm (resetEpisode in finally).

Run:
  PYTHONPATH=src AIUTOPIA_ROOT=/c/Users/Carte/OneDrive/Desktop/AiUtopia \
    AIUTOPIA_DATA_DIR=/c/Users/Carte/aiutopia-data py -3.11 scripts/natural_ab_verify.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from natural_recon import PEACEFUL_CMDS, PLAYER_NAME, PORT, _p, _run, _tick, perception_ab


def main() -> int:
    _p("=" * 72)
    _p("NATURAL A/B PERCEPTION VERIFY — log-under-leaf visibility")
    _p("=" * 72)
    _p(f"port={PORT}  player={PLAYER_NAME}")

    from aiutopia.env.wrapper import AiUtopiaPettingZooEnv

    env = AiUtopiaPettingZooEnv(
        {
            "stage": 1,
            "active_roles": ["gatherer"],
            "seed_strategy": "fixed_easy",
            "py4j_ports": [PORT],
            "tick_warp": True,
            "per_worker_seed_offset": False,
            "enable_memory_writes": False,
            "aiutopia_root_per_worker": False,
            "max_episode_ticks": 1000,
        }
    )
    try:
        import numpy as np

        from natural_recon import _pos, _raw_obs

        obs, _ = env.reset(seed=1)
        agent_obs = obs.get(PLAYER_NAME, {})
        # Cold-start spawn-race guard (copied from capture_gatherer_obs_fixture.py):
        # a freshly-launched server may /tp the agent before the Carpet player
        # exists, leaving it off-arena. Re-reset until it lands in the ±30 box.
        for attempt in range(4):
            p = list(agent_obs.get("position", [0, 0, 0]))
            if abs(p[0] - 64.0) < 30 and p[1] >= 60:
                break
            _p(f"[reset] attempt {attempt}: agent at {p} (not arena) — re-resetting")
            obs, _ = env.reset(seed=1)
            agent_obs = obs.get(PLAYER_NAME, {})
        _p(f"[reset] agent settled at pos={_pos(_raw_obs(env))}")
        for c in PEACEFUL_CMDS:
            _run(env, c)
        _tick(env, 2)
        # Diagnostic: where is the agent, and what are the nearest-row offsets
        # AFTER the whole-arena clear? (resolves baseline-saturation confounds.)
        from natural_recon import _clear_whole_arena, _nearest_rows

        _clear_whole_arena(env)
        _tick(env, 3)
        diag = _raw_obs(env)
        rows = _nearest_rows(diag)
        _p(f"[diag] post-clear agent pos={_pos(diag)}")
        _p(f"[diag] post-clear top nearest rows (dx/16,dy/8,dz/16,...): {np.round(rows[:4], 3).tolist()}")
        result = perception_ab(env)
        _p("")
        _p("RESULT (JSON):")
        print(json.dumps(result, indent=2))
        leaf_seen = result.get("hypothesis_leafed", {}).get("seen")
        ctrl_seen = result.get("control_bare_log", {}).get("seen")
        _p("")
        _p(
            f"  control_seen={ctrl_seen}  leafed_seen={leaf_seen}  "
            f"=> log-under-leaf {'VISIBLE (FIX WORKS)' if leaf_seen else 'INVISIBLE (bug present)'}"
        )
        return 0
    finally:
        try:
            env.bridge.reset_episode(PLAYER_NAME, 1)
            _p("[cleanup] resetEpisode — instance left warm")
        except Exception:
            _p("[cleanup] resetEpisode failed (non-fatal)")
        env.close()


if __name__ == "__main__":
    sys.exit(main())
