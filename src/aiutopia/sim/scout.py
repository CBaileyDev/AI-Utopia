"""FrontierScout: a Level-1 partial-information frontier explorer for the gatherer
fast-sim — the decisive artifact for the "Fork A" architecture test.

The scout emits a directional BEARING toward likely-unexplored (and therefore
likely-resource-bearing) regions from PARTIAL observation ONLY. It is the
no-oracle replacement for the ground-truth ``resource_bearing_cue`` path in
``obs_adapter.build_gatherer_obs``: where the oracle reads ``SimWorld.logs`` /
``log_alive`` to point straight at the nearest alive log (even beyond perception),
the scout knows ONLY what the agent has perceived and points toward the frontier
of the known region (classic Wavefront Frontier Detection).

CRITICAL INVARIANT (the entire validity of the Fork-A experiment):
  The scout MUST derive ``observed`` / ``resource_seen`` / frontiers ONLY from what
  is fed in via ``observe()`` and ``observe_resources()``. It MUST NEVER read
  ``SimWorld.logs``, ``SimWorld.log_alive``, or any other ground-truth world field.
  It takes no ``world`` reference at all. Violating this turns the scout back into
  an oracle and silently invalidates the held-out-geometry discriminator.

Representation: sparse occupancy via sets of integer ``(x, z)`` world cells (the
arena is small but the API stays O(observed-boundary), not O(arena)). ``observed``
is every cell the agent has perceived; ``resource_seen`` is the subset where a
resource column was reported (kept for future scoring — not yet used in the
bearing score, which is pure frontier-distance/size WFD).

Determinism: the sim forbids ``Math.random`` / ``Date`` (see world.py). Every step
here is order-free or explicitly sorted — connected-component selection breaks ties
by ``(-score, centroid_x, centroid_z)`` so the bearing is seed-reproducible.

IMPORT-LIGHT: stdlib (``collections``, ``math``) + the ``GRID_RADIUS`` constant
from the sibling ``obs_adapter`` (single source of truth for the perception window;
no cycle — ``obs_adapter`` does not import ``scout``). Never numpy/torch/py4j/chroma.
"""

from __future__ import annotations

import math
from collections import deque

from aiutopia.sim.obs_adapter import GRID_RADIUS

__all__ = ["FrontierScout", "SweepScout"]

# 8-connectivity offsets for connected-component grouping (WFD groups adjacent
# frontier cells, including diagonals).
_NEIGHBORS_8: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# 4-connectivity offsets for the frontier test (a cell is a frontier iff >=1 of its
# 4-neighbours is NOT observed — the boundary of the known region).
_NEIGHBORS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class FrontierScout:
    """Partial-information frontier explorer.

    See module docstring for the load-bearing invariant: this class only ever
    learns about the world through ``observe()`` / ``observe_resources()``. It
    never touches ground truth.
    """

    def __init__(self) -> None:
        # Cells (x,z) the agent has perceived. Sparse set keyed by world coords.
        self.observed: set[tuple[int, int]] = set()
        # Subset of observed cells that reported a resource column (kept for
        # future scoring; not part of the current WFD bearing score).
        self.resource_seen: set[tuple[int, int]] = set()

    def observe(self, bx: int, bz: int) -> None:
        """Mark every cell in the agent's 32x32 perception window as observed.

        The window mirrors ``obs_adapter.gatherer_nearest_columns``:
        ``[bx-GRID_RADIUS, bx+GRID_RADIUS) x [bz-GRID_RADIUS, bz+GRID_RADIUS)``
        (half-open, ``+GRID_RADIUS`` exclusive). ``bx, bz`` are ``floor(agent x),
        floor(agent z)``. Derives nothing from ground truth — only the window
        geometry around the agent's reported position.
        """
        bx = int(bx)
        bz = int(bz)
        for dx in range(-GRID_RADIUS, GRID_RADIUS):
            for dz in range(-GRID_RADIUS, GRID_RADIUS):
                self.observed.add((bx + dx, bz + dz))

    def observe_resources(self, cells: list[tuple[int, int]]) -> None:
        """Record perceived resource cells (absolute world (x,z)).

        Fed from ``gatherer_nearest_columns`` (the SCAN_RADIUS subset the agent
        actually perceives). These are already inside the perception window, so
        they are also marked observed for consistency. Kept for future
        resource-weighted scoring; the current bearing uses frontiers only.
        """
        for x, z in cells:
            cell = (int(x), int(z))
            self.resource_seen.add(cell)
            self.observed.add(cell)

    def frontiers(self) -> set[tuple[int, int]]:
        """Observed cells with >=1 of their 4-neighbours NOT observed.

        This is the boundary of the known region — the frontier between explored
        and unexplored space. For any non-empty ``observed`` on an unbounded grid
        this is non-empty (the outermost ring always borders unobserved cells), so
        ``bearing()`` returns ``None`` only when ``observed`` is empty.
        """
        observed = self.observed
        out: set[tuple[int, int]] = set()
        for cell in observed:
            cx, cz = cell
            for ddx, ddz in _NEIGHBORS_4:
                if (cx + ddx, cz + ddz) not in observed:
                    out.add(cell)
                    break
        return out

    def _components(
        self, frontier: set[tuple[int, int]]
    ) -> list[set[tuple[int, int]]]:
        """Group frontier cells into 8-connected components via BFS (WFD).

        Deterministic: we BFS from cells in a sorted seed order and each
        component is a set, so the grouping is independent of set-iteration order.
        """
        unvisited = set(frontier)
        components: list[set[tuple[int, int]]] = []
        # Sorted seed order so the component construction is seed-reproducible.
        for seed in sorted(frontier):
            if seed not in unvisited:
                continue
            comp: set[tuple[int, int]] = set()
            queue: deque[tuple[int, int]] = deque((seed,))
            unvisited.discard(seed)
            while queue:
                cx, cz = queue.popleft()
                comp.add((cx, cz))
                for ddx, ddz in _NEIGHBORS_8:
                    nb = (cx + ddx, cz + ddz)
                    if nb in unvisited:
                        unvisited.discard(nb)
                        queue.append(nb)
            components.append(comp)
        return components

    def bearing(self, bx: int, bz: int) -> tuple[float, float, float] | None:
        """Horizontal unit bearing + distance to the best frontier component.

        Groups frontier cells into 8-connected components (WFD), scores each
        ``size / (dist_from_agent_to_centroid + 1)``, picks the highest-scoring
        (deterministic tie-break ``(-score, centroid_x, centroid_z)``), and returns
        ``(unit_dx, unit_dz, euclidean_dist_to_centroid)`` — a HORIZONTAL (x,z) unit
        vector. ``bx, bz`` are ``floor(agent x), floor(agent z)``.

        Returns ``None`` when there are no frontiers (i.e. ``observed`` empty), or
        when the chosen centroid coincides with the agent (degenerate zero-length
        direction — guarded to avoid a NaN unit vector).
        """
        frontier = self.frontiers()
        if not frontier:
            return None
        ax = float(bx)
        az = float(bz)

        scored: list[tuple[float, float, float, float, float]] = []
        for comp in self._components(frontier):
            size = len(comp)
            sx = sum(c[0] for c in comp)
            sz = sum(c[1] for c in comp)
            cx = sx / size
            cz = sz / size
            dist = math.hypot(cx - ax, cz - az)
            score = size / (dist + 1.0)
            scored.append((score, cx, cz, dist, float(size)))

        # Deterministic selection: highest score, then smallest centroid_x, then
        # smallest centroid_z. sorted() before picking so set-iteration order in
        # _components can never leak into the chosen bearing.
        scored.sort(key=lambda r: (-r[0], r[1], r[2]))
        _score, cx, cz, dist, _size = scored[0]

        ddx = cx - ax
        ddz = cz - az
        norm = math.hypot(ddx, ddz)
        if norm < 1e-9:
            # Degenerate: centroid on top of the agent -> no meaningful direction.
            return None
        return (ddx / norm, ddz / norm, dist)


class SweepScout:
    """A COMMITTED ring-sweep producer (partial-info, same interface as FrontierScout).

    The greedy WFD ``FrontierScout`` dithers (it re-picks the nearest frontier each
    step and never commits to traverse outward), so on the omni arena it only
    modestly beats a fixed heading. ``SweepScout`` instead visits a fixed ring of
    waypoints at radius ``R`` around the SPAWN point, one compass sector at a time,
    so it systematically sweeps ALL directions — guaranteed to bring a cluster at
    radius ~``R`` within the 16-block perception regardless of its bearing.

    It uses only the agent's reported position (captured spawn + current pos); it
    reads NO ground truth (same invariant as FrontierScout). ``observe`` /
    ``observe_resources`` exist for interface parity; only the first ``observe``
    (or ``bearing``) call is used, to capture the spawn origin.

    n_dirs=12 at R=28: adjacent ring points are ~14.5 blocks apart (< 2*16
    perception), so the swept ring has no perception gaps; a cluster anywhere on
    the ring is seen at some waypoint.
    """

    def __init__(self, n_dirs: int = 12) -> None:
        # Search directions, evenly spaced over the full circle.
        self._dirs: list[tuple[float, float]] = [
            (math.cos(2.0 * math.pi * k / n_dirs), math.sin(2.0 * math.pi * k / n_dirs))
            for k in range(n_dirs)
        ]
        self._idx = 0
        self._phase_out = True  # True: hop OUT in dirs[idx]; False: hop BACK to spawn
        self._spawn: tuple[int, int] | None = None
        # Present for interface parity with FrontierScout (unused by the sweep).
        self.observed: set[tuple[int, int]] = set()
        self.resource_seen: set[tuple[int, int]] = set()

    def _ensure_spawn(self, bx: int, bz: int) -> None:
        if self._spawn is None:
            self._spawn = (int(bx), int(bz))

    def observe(self, bx: int, bz: int) -> None:
        self._ensure_spawn(bx, bz)

    def observe_resources(self, cells: list[tuple[int, int]]) -> None:
        return None

    def bearing(self, bx: int, bz: int) -> tuple[float, float, float] | None:
        """OUT-and-BACK sweep matched to the NAVIGATE dynamics (each dispatch moves
        a FIXED ~MAX_NAV_RANGE=32 blocks in the bearing direction, NOT "to a point").

        Each blind step is one 32-block hop, so we alternate: OUT (hop 32 in
        ``dirs[idx]`` → lands at radius ~32, where a cluster at radius 24-30 in
        roughly that direction enters the 16-block perception), then BACK (hop
        toward spawn), then rotate to the next direction. Radius-32 out-hops stay
        inside the omni arena (arena_half~40), so the agent never pins on the wall
        (the failure mode of a waypoint-tracking sweep). 12 directions ⇒ every
        bearing is within 15° of some hop ⇒ a cluster in ANY direction is swept."""
        self._ensure_spawn(bx, bz)
        ax, az = float(bx), float(bz)
        assert self._spawn is not None
        if self._phase_out:
            dx, dz = self._dirs[self._idx]
            self._phase_out = False
            return (dx, dz, 32.0)
        # BACK toward spawn, then rotate to the next direction.
        ddx = self._spawn[0] - ax
        ddz = self._spawn[1] - az
        self._phase_out = True
        self._idx = (self._idx + 1) % len(self._dirs)
        norm = math.hypot(ddx, ddz)
        if norm < 1e-9:
            dx, dz = self._dirs[self._idx]
            return (dx, dz, 32.0)
        return (ddx / norm, ddz / norm, norm)
