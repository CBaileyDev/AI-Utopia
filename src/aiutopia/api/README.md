# AI Utopia GUI backend (FastAPI)

Bridges the Tauri desktop GUI to the existing Python systems. Implements
`gui/API_CONTRACT.md` (all routes under `/api`, server on `http://127.0.0.1:8777`,
CORS open for the Tauri webview).

## Launch

```bash
# bash
PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 -m aiutopia.api
#   or
PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 scripts/run_api.py

# PowerShell
$env:PYTHONPATH='src'; $env:PYTHONIOENCODING='utf-8'; py -3.11 -m aiutopia.api
```

Env overrides: `AIUTOPIA_API_HOST` (default `127.0.0.1`), `AIUTOPIA_API_PORT`
(default `8777`). Point `AIUTOPIA_ROOT` at the repo so `runs/`, `identity.db`,
etc. resolve (the operational value is the repo path).

## Design invariants

- **Lazy heavy imports.** `chromadb` / `py4j` / `ray` / `torch` are imported
  *inside* route handlers, never at module top. The server boots in <1s and a
  dead Minecraft server only fails the agent routes — `/api/health` and
  `/api/training/*` work with no Minecraft and no heavy deps loaded.
- **Never an HTML 500.** A global exception handler returns
  `{ok:false, error:str}` JSON (HTTP 200) for any unhandled error; per-route
  try/except handles expected failures (bridge down, db missing).
- **One training run at a time.** The server tracks the subprocess it launched
  (`Popen` + a `Research/train-gui-*.pid` file). `/start` rejects if one is
  alive; `/stop` terminates it. Liveness is `Popen.poll()` — externally-started
  runs are reported `done` (Windows can't reliably probe a foreign PID).

## Routes (real vs stub)

| Route | Source | Notes |
|-------|--------|-------|
| `GET /api/health` | real | probes FabricBridge on the production Py4J port (25100); `offline` if down |
| `GET /api/agents` | real | identity.db living agents + best-effort live x/z/health/hunger join (by `agent_name`); `[]` if none. `rewards` is a stub `0.0` (not tracked in identity.db) |
| `POST /api/agents/spawn` | real | identity row + Chroma collections + best-effort Carpet spawn; persists identity even if Minecraft is down (reports `carpet_spawned`) |
| `POST /api/agents/{uuid}/kill` | real | `IdentityService.record_death` (matches the CLI; no Carpet `/player kill` exists in the bridge) |
| `GET /api/training/runs` | real | scans `runs/aiutopia_*/PPO_*/progress.csv` |
| `GET /api/training/status` | real | parses the last rows of the active/newest run's `progress.csv` |
| `POST /api/training/start` | real | launches `scripts/train.py` as a subprocess (default `--backend sim`) |
| `POST /api/training/stop` | real | terminates the tracked subprocess |
| `GET /api/rewards` | real | `aiutopia.env.reward.load_reward_config()` (defaults + on-disk overlay) |
| `PUT /api/rewards` | real | deep-merges the partial body over the current config and writes the overlay JSON |
| `GET /api/logs` | real-ish | tails the GUI job's log, else the newest `Research/train*.log` |

### Metrics mapping (`progress.csv` columns)

- `return_mean` ← `env_runners/episode_return_mean`
- `entropy` ← `learners/gatherer_policy/entropy`
- `kl` ← `learners/gatherer_policy/mean_kl_loss`
- `iter` ← `training_iteration`
- `sps` ← `env_runners/num_env_steps_sampled_lifetime_throughput/throughput_since_last_reduce`
- `clipfrac` / `term_rate` → **`null`**: no PPO clip-fraction / termination-rate
  column exists in RLlib's `progress.csv`, so they are reported as `null` rather
  than fabricated.

## Reward config externalization

`src/aiutopia/env/reward.py` keeps the parity-critical constants as literals and
exposes a loader (`load_reward_config`) that deep-merges an optional on-disk JSON
overlay over them. **With no overlay file the merged values are byte-identical to
the historical constants** — the reward + parity suites stay green.

Overlay file resolution (first match wins):
1. `$AIUTOPIA_REWARD_CONFIG` (explicit path; the GUI's write target)
2. `$AIUTOPIA_ROOT/config/rewards.json`
3. `<repo>/config/rewards.json`

`config/rewards.default.json` is a committed *reference* of the defaults; it is
**not** `rewards.json`, so its presence does not activate any overlay. Values are
bound at import time, so a `PUT /api/rewards` takes effect on the **next** process
/ training start (no hot-reload — that would re-introduce the parity risk).
