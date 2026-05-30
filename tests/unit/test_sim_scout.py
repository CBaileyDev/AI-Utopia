"""Fork-A FrontierScout tests: the partial-information scout that replaces the
ground-truth bearing oracle.

The decisive test is the PARTIAL-INFO DISCRIMINATOR: a scout stepped near
cluster A must NOT have any beyond-perception cluster-B cell in its ``observed``
set — proving it cannot see B until the agent gets within GRID_RADIUS blocks.
B's location is derived from the world ONLY inside the test (for the assertion);
the scout itself never reads ground truth.

Plus: bearing() shape (unit horizontal vector + finite dist, or None on an empty
scout), WFD component grouping, per-cell frontier shrink, and obs back-compat
(bearing_override=None is byte-identical to before for a fixed world).

IMPORT-LIGHT: drives only the sim package + a couple of env leaf helpers (no
chroma / py4j / torch).
"""

from __future__ import annotations

import math

import numpy as np

from aiutopia.sim.obs_adapter import GRID_RADIUS, build_gatherer_obs
from aiutopia.sim.scout import FrontierScout
from aiutopia.sim.sim_env import AiUtopiaSimEnv
from aiutopia.sim.world import SimWorld


def _act(skill: int) -> dict:
    return {
        "skill_type": skill,
        "target_class": 0,
        "spatial_param": np.zeros(3, np.float32),
        "scalar_param": np.array([1.0], np.float32),
        "comm_payload": np.zeros(128, np.float32),
        "should_broadcast": 0,
        "comm_target_mask": np.zeros(4, np.int8),
    }


def _spawn_window(bx: int, bz: int) -> set[tuple[int, int]]:
    """The exact 32x32 half-open perception window the scout marks for an agent at
    floor-pos (bx,bz) — recomputed INDEPENDENTLY of the scout for cross-checking."""
    return {
        (bx + dx, bz + dz)
        for dx in range(-GRID_RADIUS, GRID_RADIUS)
        for dz in range(-GRID_RADIUS, GRID_RADIUS)
    }


# ── PARTIAL-INFO DISCRIMINATOR (the key test) ───────────────────────────────────
def test_scout_cannot_see_cluster_b_while_beyond_perception():
    """Step the agent near cluster A WITHOUT moving toward B (WAIT only), then prove
    the scout's observed set contains NO beyond-perception cluster-B cell — i.e. the
    scout literally cannot perceive B until the agent gets within GRID_RADIUS. B's
    location comes from the world ONLY here (the assertion), never from the scout."""
    env = AiUtopiaSimEnv(
        {
            "active_roles": ["gatherer"],
            "decision_core": True,
            "scout_mode": "real",
            "arena_mode": "clusters",
            "arena_half": 34.0,
            "max_episode_ticks": 200,
        }
    )
    env.reset(seed=90001)
    # Stay put: WAIT keeps the agent at spawn so observed == exactly the spawn window.
    for _ in range(5):
        env.step({"gatherer_0": _act(4)})

    world = env.worlds["gatherer_0"]
    scout = env._scouts["gatherer_0"]
    bx = int(math.floor(float(world.agent_pos[0])))
    bz = int(math.floor(float(world.agent_pos[2])))

    # Cluster construction order (world._cluster_bases): cluster A first (8 bases x
    # TRUNK_H), then cluster B -> logs[32:64] are B. Derive B's cells from the world.
    b_cells = {
        (int(world.logs[i][0]), int(world.logs[i][2]))
        for i in range(32, world.logs.shape[0])
    }

    def _beyond(cell: tuple[int, int]) -> bool:
        dx = cell[0] - bx
        dz = cell[1] - bz
        return (
            dx < -GRID_RADIUS
            or dx >= GRID_RADIUS
            or dz < -GRID_RADIUS
            or dz >= GRID_RADIUS
        )

    b_beyond = {c for c in b_cells if _beyond(c)}

    # Non-vacuous: there must be at least one B cell genuinely beyond perception.
    assert b_beyond, "test is vacuous — all cluster-B cells fell inside perception"
    # The scout must NOT have observed any beyond-perception B cell.
    assert scout.observed.isdisjoint(b_beyond)
    # Strongest no-leak assertion: observed is EXACTLY the recomputed spawn window —
    # the scout learned nothing the agent could not perceive.
    assert scout.observed == _spawn_window(bx, bz)


def test_scout_sees_b_only_after_walking_within_perception():
    """Positive control for the discriminator: a scout that is fed an observe()
    window centered ON cluster B does mark B cells — confirming the disjointness
    above is about PERCEPTION, not a scout that can never see B at all."""
    world = SimWorld()
    world.reset(90001, arena_mode="clusters")
    b_cells = {
        (int(world.logs[i][0]), int(world.logs[i][2]))
        for i in range(32, world.logs.shape[0])
    }
    # Centroid of cluster B; observe a window there (as if the agent walked over).
    cx = round(sum(c[0] for c in b_cells) / len(b_cells))
    cz = round(sum(c[1] for c in b_cells) / len(b_cells))
    scout = FrontierScout()
    scout.observe(cx, cz)
    assert scout.observed & b_cells, "scout should perceive B when standing in it"


# ── bearing() shape ──────────────────────────────────────────────────────────────
def test_bearing_none_on_empty_scout():
    """No observations -> no frontiers -> bearing is None."""
    scout = FrontierScout()
    assert scout.frontiers() == set()
    assert scout.bearing(0, 0) is None


def test_bearing_is_unit_horizontal_with_finite_dist():
    """After observing an ASYMMETRIC region the bearing is a horizontal unit vector
    (hypot ~1) with a finite, positive distance to the frontier centroid."""
    scout = FrontierScout()
    scout.observe(0, 0)
    scout.observe(40, 0)  # second window far in +x -> a real, non-degenerate direction
    b = scout.bearing(0, 0)
    assert b is not None
    udx, udz, dist = b
    assert math.isfinite(udx) and math.isfinite(udz) and math.isfinite(dist)
    assert abs(math.hypot(udx, udz) - 1.0) < 1e-6
    assert dist > 0.0


# ── WFD component grouping ───────────────────────────────────────────────────────
def test_wfd_groups_adjacent_cells_into_one_component():
    """Two disjoint observed blobs -> the frontier splits into 2 connected
    components; a single blob -> 1 component (8-connectivity)."""
    scout = FrontierScout()
    scout.observe(0, 0)
    one = scout._components(scout.frontiers())
    assert len(one) == 1  # a single contiguous frontier ring

    scout2 = FrontierScout()
    scout2.observe(0, 0)
    scout2.observe(100, 100)  # far away -> separate frontier ring
    two = scout2._components(scout2.frontiers())
    assert len(two) == 2


def test_components_are_deterministic():
    """Same observations -> same component count + same chosen bearing (no
    set-iteration leakage)."""
    a = FrontierScout()
    b = FrontierScout()
    for s in (a, b):
        s.observe(0, 0)
        s.observe(40, 5)
    assert a.bearing(0, 0) == b.bearing(0, 0)


# ── per-cell frontier shrink ─────────────────────────────────────────────────────
def test_frontier_cell_leaves_set_once_surrounded():
    """A cell that is a frontier (has an unobserved neighbour) leaves the frontier
    set once all its 4-neighbours become observed — the per-cell shrink the WFD
    relies on (total count is NOT monotonic, so we assert per-cell)."""
    scout = FrontierScout()
    # A single observed cell is a frontier (all 4 neighbours unobserved).
    scout.observed.add((0, 0))
    assert (0, 0) in scout.frontiers()
    # Observe all 4-neighbours -> (0,0) is now interior, no longer a frontier.
    for dx, dz in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        scout.observed.add((dx, dz))
    assert (0, 0) not in scout.frontiers()


def test_observe_is_idempotent():
    """Re-observing the same window does not change observed (sparse-set semantics)."""
    scout = FrontierScout()
    scout.observe(10, -10)
    snapshot = set(scout.observed)
    scout.observe(10, -10)
    assert scout.observed == snapshot


# ── obs back-compat (the entire point of branch ordering) ────────────────────────
def _obs_bytes_equal(a: dict, b: dict) -> bool:
    """Recursive byte-equality over the obs dict (action_mask is a nested dict)."""
    if a.keys() != b.keys():
        return False
    for k in a:
        av, bv = a[k], b[k]
        if isinstance(av, dict) or isinstance(bv, dict):
            if not (isinstance(av, dict) and isinstance(bv, dict) and _obs_bytes_equal(av, bv)):
                return False
        elif not np.array_equal(np.asarray(av), np.asarray(bv)):
            return False
    return True


def test_bearing_override_none_is_byte_identical_oracle():
    """bearing_override=None preserves EXACTLY the prior oracle behaviour for a
    fixed world (the resource_bearing_cue=True path)."""
    world = SimWorld()
    world.reset(1, arena_mode="clusters")
    before = build_gatherer_obs(world, resource_bearing_cue=True)
    after = build_gatherer_obs(world, resource_bearing_cue=True, bearing_override=None)
    assert _obs_bytes_equal(before, after)
    # Oracle is active -> g_hostiles row 0 is non-zero in both.
    assert np.asarray(after["g_hostiles_nearby"])[0].any()


def test_bearing_override_none_is_byte_identical_off():
    """bearing_override=None with the cue off preserves the all-zeros (off) path."""
    world = SimWorld()
    world.reset(1, arena_mode="clusters")
    before = build_gatherer_obs(world, resource_bearing_cue=False)
    after = build_gatherer_obs(world, resource_bearing_cue=False, bearing_override=None)
    assert _obs_bytes_equal(before, after)
    assert not np.asarray(after["g_hostiles_nearby"]).any()


def test_bearing_override_takes_precedence_over_cue():
    """A supplied override is written verbatim and the ground-truth oracle is NOT
    consulted (precedence: override > resource_bearing_cue)."""
    world = SimWorld()
    world.reset(1, arena_mode="clusters")
    obs = build_gatherer_obs(
        world, resource_bearing_cue=True, bearing_override=(1.0, 0.0, 32.0)
    )
    row = np.asarray(obs["g_hostiles_nearby"])[0]
    assert row[0] == 1.0 and row[1] == 0.0
    assert abs(float(row[2]) - 0.5) < 1e-6  # 32/64
    assert row[3] == 1.0
