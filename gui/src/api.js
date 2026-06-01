/* AI Utopia — backend fetch client.
 *
 * One thin, typed-ish wrapper per route in gui/API_CONTRACT.md. Every call:
 *   - targets the FastAPI backend at API_BASE (default 127.0.0.1:8777/api),
 *   - aborts after a short timeout (so the UI never hangs on a dead backend),
 *   - parses JSON and returns it,
 *   - THROWS on any network / HTTP / timeout error.
 *
 * Throwing is deliberate: callers (useResource in useApi.js) catch the throw and
 * fall back to mock data + an "offline" flag, so the GUI stays alive offline.
 *
 * Works identically in a plain browser (npm run dev) and inside the Tauri v2
 * webview: we use the webview's native `fetch` to a remote http://127.0.0.1 URL,
 * which Tauri allows without the @tauri-apps/plugin-http permission. CSP is
 * `null` in tauri.conf.json so connect-src is unrestricted.
 */

// Overridable at runtime via `window.__AIUTOPIA_API_BASE__` (set before the app
// boots) for future packaging flexibility; otherwise the localhost default.
export const API_BASE =
  (typeof window !== 'undefined' && window.__AIUTOPIA_API_BASE__) ||
  'http://127.0.0.1:8777/api';

const DEFAULT_TIMEOUT = 4000;

/** Low-level fetch: JSON in/out, AbortController timeout, throws on !ok. */
async function request(path, { method = 'GET', body, timeout = DEFAULT_TIMEOUT } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method,
      signal: ctrl.signal,
      headers: body != null ? { 'Content-Type': 'application/json' } : undefined,
      body: body != null ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    // Network failure, DNS, or abort (timeout) — normalize to one error type.
    throw new ApiError(e.name === 'AbortError' ? 'timeout' : 'network', path, e);
  } finally {
    clearTimeout(timer);
  }
  if (!res.ok) {
    let detail;
    try { detail = await res.json(); } catch { /* non-JSON body */ }
    throw new ApiError(`http ${res.status}`, path, detail);
  }
  // 204 / empty body tolerance
  const text = await res.text();
  return text ? JSON.parse(text) : null;
}

export class ApiError extends Error {
  constructor(kind, path, cause) {
    super(`API ${kind} @ ${path}`);
    this.name = 'ApiError';
    this.kind = kind;
    this.path = path;
    this.cause = cause;
    // Surface a backend-provided { error } message when present.
    this.detail = cause && typeof cause === 'object' ? cause.error || cause.detail : undefined;
  }
}

/* ---------------------------------------------------------------- Health */
export const getHealth = () => request('/health');

/* ---------------------------------------------------------------- Agents */
export const getAgents = () => request('/agents');
export const spawnAgent = (role, name) =>
  request('/agents/spawn', { method: 'POST', body: name ? { role, name } : { role } });
export const killAgent = (uuid, cause) =>
  request(`/agents/${encodeURIComponent(uuid)}/kill`, { method: 'POST', body: cause ? { cause } : {} });

/* -------------------------------------------------------------- Training */
export const getTrainingRuns = () => request('/training/runs');
// status polls frequently; give it a slightly tighter timeout.
export const getTrainingStatus = () => request('/training/status', { timeout: 3000 });
export const startTraining = (opts) =>
  request('/training/start', { method: 'POST', body: opts, timeout: 8000 });
export const stopTraining = () => request('/training/stop', { method: 'POST', timeout: 8000 });

/* --------------------------------------------------------------- Rewards */
export const getRewards = () => request('/rewards');
export const putRewards = (config) => request('/rewards', { method: 'PUT', body: config, timeout: 6000 });

/* ------------------------------------------------------------------ Logs */
export const getLogs = (tail = 200) => request(`/logs?tail=${tail}`);
