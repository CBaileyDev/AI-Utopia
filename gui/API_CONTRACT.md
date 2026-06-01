# AI Utopia GUI <-> Backend API contract (v1)

FastAPI backend at http://127.0.0.1:8777, all routes under /api. JSON in/out. CORS open to the Tauri webview.
Frontend (gui/src/api.js) calls these; falls back to mock data when the backend is unreachable so the app still renders offline.

## Health / system
GET /api/health -> { bridge: "online"|"offline", py4j_port: 25100, instances: int, mc_version: "1.21.1", server_time: iso }

## Agents (identity.db + FabricBridge; real)
GET  /api/agents -> [ { id, name, role, status, uuid, skin, born, x, z, rewards, health, hunger } ]   # [] if none
POST /api/agents/spawn  body { role: "gatherer"|"builder"|"farmer"|"defender", name?: str } -> { ok, agent } | { ok:false, error }
POST /api/agents/{uuid}/kill  body { cause?: str } -> { ok }

## Training (scan runs/*, parse progress.csv, manage process)
GET  /api/training/runs -> [ { run_id, seed, backend, iters, last_return, status: "running"|"done"|"errored", path } ]
GET  /api/training/status -> { running: bool, run_id?, backend?, iter, max_iters, sps?, metrics: { return_mean, entropy, kl, clipfrac, term_rate? }, history: [ { iter, return_mean, entropy, kl } ] }
POST /api/training/start body { backend: "sim"|"real", iters: int, num_envs?: int, entropy_coeff?, spawn_jitter?, approach_shaping?, force_masked_spawn? } -> { ok, pid, run_id }
POST /api/training/stop -> { ok }

## Rewards (externalized config; real read/write)
GET /api/rewards -> { log_value: {item:float}, pbrs: { gamma, time_penalty, death_penalty }, role_task_items: {role:[item]}, role_caps: {...} }
PUT /api/rewards body (same shape, partial allowed) -> { ok, saved_path }   # writes the config file the env loads

## Logs
GET /api/logs?tail=200 -> [ { ts, type: "AGENT"|"TRAIN"|"SYSTEM"|"CHAT", message } ]   # tail latest train log + bridge events; mock-ish ok

## Notes
- Heavy imports (chroma/py4j/ray) lazy-loaded per route so the server starts instantly and a dead Minecraft server only fails the agent routes, not the whole API.
- Every route returns a clean JSON error (never a 500 HTML page) so the frontend can show a friendly state.
