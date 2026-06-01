/* AI Utopia — data-layer hooks.
 *
 * useResource(fetcher, { fallback, pollMs, deps, enabled, transform })
 *   Wraps an api.js call. Returns { data, loading, error, online, refetch }.
 *
 *   - On success: data = (transform ? transform(raw) : raw), online = true.
 *   - On throw (backend unreachable / HTTP error): online = false, error set,
 *     and data falls back to `fallback` (a mockData export) so the GUI still
 *     renders a believable view instead of breaking.
 *   - `pollMs`: re-fetch on an interval (health ~5s, training status ~2s).
 *   - `enabled`: pause polling/fetching (e.g. only poll status on the Training tab).
 *
 * The online/offline distinction is the seam the whole "graceful offline" story
 * hangs on: pages show real data + intentional empty states when online, and a
 * subtle "sample data" pill over mock when offline.
 *
 * IMPORTANT (polling correctness): callers pass inline fallbacks/fetchers (e.g.
 * `fallback: []`) that are a fresh reference every render. The effect that sets
 * up the poll interval therefore must NOT depend on those identities, or it
 * would tear down and recreate the interval on every render and stack a flood
 * of timers. We keep fetcher/transform/fallback in refs and key the effect only
 * on the primitive controls (enabled, pollMs) plus an explicit `deps` array.
 */
import { useState, useEffect, useRef, useCallback } from 'react';

export function useResource(fetcher, { fallback, pollMs, deps = [], enabled = true, transform } = {}) {
  const [data, setData] = useState(fallback);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [online, setOnline] = useState(false);

  // Keep latest fetcher/transform/fallback without retriggering the effect.
  const fetcherRef = useRef(fetcher);
  const transformRef = useRef(transform);
  const fallbackRef = useRef(fallback);
  fetcherRef.current = fetcher;
  transformRef.current = transform;
  fallbackRef.current = fallback;

  const aliveRef = useRef(true);

  // Stable across renders — reads everything mutable from refs.
  const run = useCallback(async () => {
    try {
      const raw = await fetcherRef.current();
      if (!aliveRef.current) return;
      const next = transformRef.current ? transformRef.current(raw) : raw;
      setData(next);
      setOnline(true);
      setError(null);
    } catch (e) {
      if (!aliveRef.current) return;
      setOnline(false);
      setError(e);
      setData((prev) => (fallbackRef.current !== undefined ? fallbackRef.current : prev));
    } finally {
      if (aliveRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    aliveRef.current = true;
    if (!enabled) {
      setLoading(false);
      return () => { aliveRef.current = false; };
    }
    setLoading(true);
    run();
    let id;
    if (pollMs) id = setInterval(run, pollMs);
    return () => {
      aliveRef.current = false;
      if (id) clearInterval(id);
    };
    // Only re-subscribe on primitive control changes + explicit deps. `run` is
    // stable; fallback/fetcher identity intentionally excluded (see header).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, pollMs, run, ...deps]);

  return { data, loading, error, online, refetch: run };
}
