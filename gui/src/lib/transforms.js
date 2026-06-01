/* AI Utopia — live-API → mock-shape adapters.
 *
 * The components were written against mockData.js shapes and must stay
 * pixel-faithful, so rather than touch every render site we normalize each live
 * response back into the shape the component already consumes. Fields the live
 * source genuinely lacks (policyLoss/valueLoss/duration/status on training
 * history; clipfrac/term_rate on metrics) are passed through as null and the
 * components render an em-dash for them.
 */

const KNOWN_ROLES = new Set(['gatherer', 'builder', 'farmer', 'defender']);
const KNOWN_STATUS = new Set(['alive', 'training', 'idle', 'offline']);

/** Logs: {ts(ISO), type, message} -> {id, timestamp:'HH:MM:SS', type, message}. */
export function adaptLogs(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map((l, i) => {
    let timestamp = '';
    try {
      timestamp = new Date(l.ts).toLocaleTimeString('en-US', { hour12: false });
    } catch {
      timestamp = String(l.ts ?? '').slice(11, 19);
    }
    return { id: `${l.ts ?? i}-${i}`, timestamp, type: l.type || 'SYSTEM', message: l.message ?? '' };
  });
}

/** Agents: contract shape already matches mock; just sanitize role/status so
 *  Avatar (roleMeta[role].icon) and StatusDot (statusColor[status]) never throw. */
export function adaptAgents(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map((a, i) => ({
    id: a.id ?? a.uuid ?? String(i),
    name: a.name ?? 'agent',
    role: KNOWN_ROLES.has(a.role) ? a.role : 'gatherer',
    status: KNOWN_STATUS.has(a.status) ? a.status : 'idle',
    uuid: a.uuid ?? a.id ?? '',
    skin: a.skin ?? 'Default Villager',
    born: a.born ?? null,
    x: a.x ?? 0,
    z: a.z ?? 0,
    rewards: typeof a.rewards === 'number' ? a.rewards : 0,
    health: typeof a.health === 'number' ? a.health : 20,
    hunger: typeof a.hunger === 'number' ? a.hunger : 20,
  }));
}

/** Training history -> epoch-table rows. Missing cols stay null (rendered '—'). */
export function adaptEpochs(history) {
  if (!Array.isArray(history)) return [];
  // newest first to match the mock table's default desc sort feel
  return [...history]
    .map((h) => ({
      epoch: h.iter,
      meanReward: h.return_mean,
      entropy: h.entropy,
      klDiv: h.kl,
      policyLoss: null,
      valueLoss: null,
      duration: null,
      status: null,
    }))
    .reverse();
}

/** Training history -> reward line-chart series {epoch, reward}. */
export function adaptRewardCurve(history) {
  if (!Array.isArray(history) || history.length === 0) return [];
  return history.map((h) => ({ epoch: h.iter, reward: h.return_mean }));
}

/** Pick the metrics block the Training Metrics() cards read. */
export function adaptMetrics(status) {
  const m = (status && status.metrics) || {};
  return {
    return_mean: m.return_mean ?? null,
    entropy: m.entropy ?? null,
    kl: m.kl ?? null,
    clipfrac: m.clipfrac ?? null,
    term_rate: m.term_rate ?? null,
  };
}

/** Format a maybe-null number to fixed places, em-dash on null. */
export function fmt(v, decimals = 2, prefix = '') {
  if (v == null || Number.isNaN(v)) return '—';
  return prefix + Number(v).toFixed(decimals);
}
