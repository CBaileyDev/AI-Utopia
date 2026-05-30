"""SimWorld: fast headless world state for the gatherer M1B task.

Mirrors ``WorldOps.resetEpisode`` (fabric_mod/.../bridge/WorldOps.java) exactly:
16 vertical BARE oak trunks (4 logs each, Y=65..68 = 64 logs total) on a 4x4
spaced grid, seeded +-1 (x,z) jitter per trunk, clamped to the arena, dedup-nudged
off the spawn tile and off already-placed trunks. (N21 Inc2: was a flat 8x8
single-log grid.) Trunk height 4 <= REACH so every log is ground-reachable.

IMPORT-LIGHT by contract: this module imports only ``dataclasses`` and ``numpy``
-- never chromadb / py4j / torch / sentence_transformers (verified by the focused
test and ``py -3.11 -c "import aiutopia.sim.world"``).

RNG parity (parity-critical): the real arena layout comes from Java's
``java.util.Random`` (``epRand.setSeed(seed)`` then ``epRand.nextInt(3)`` per axis).
NumPy's ``Generator(PCG64(seed))`` would produce a *different* jitter sequence for
the same seed, silently desyncing the log positions from real Minecraft and
breaking the Phase-A golden-trace fidelity gate (Task 3b, seed=1). Because the RNG
lives in THIS file, we replicate ``java.util.Random`` byte-faithfully here so the
seed=1 layout matches real MC. Task 1's tests only check structural invariants +
seed determinism, so this choice keeps them green while also fixing fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Arena constants (mirror WorldOps.resetEpisode).
# ---------------------------------------------------------------------------
SPAWN_X = 64
SPAWN_Z = -48
MIN_X, MAX_X = 48, 80
MIN_Z, MAX_Z = -64, -32
LOG_Y = 65  # N21: trunk BASE on the grass (grass top is Y=65); was 66 = floating
            # one air block above the ground. A 4-tall trunk is now Y=65..68.
GRID = 8  # legacy: 8x8 == 64 (kept so logs/log_alive default factories size to 64)

# N21 Inc2: the arena is 16 vertical BARE oak trunks (4 logs each, Y=65..68),
# replacing the flat 8x8 single-log grid. Total stays 64 (gate target unchanged).
# Trunk height 4 <= REACH (4.5) so a ground-standing agent reaches every log
# (no climbing). No leaves (collidable -> would block the real-MC walk).
TREE_GRID = 4                   # 4x4 trunk grid
TREES = TREE_GRID * TREE_GRID   # 16 trunks
TRUNK_H = 4                     # logs per trunk (16 * 4 == 64)

# Agent spawn as reported in obs-space. Java teleports to (64, 66, -48) but the
# fake player settles onto the grass top (block-top Y=65 -> feet Y=65) and obs
# reports the block-center x/z, hence [64.5, 65.0, -47.5] (confirmed by the
# n14_reward_probe trace cited in the plan). NOT the raw /tp coordinate.
SPAWN_POS = np.array([64.5, 65.0, -47.5], dtype=np.float64)


class _JavaRandom:
    """Byte-faithful port of ``java.util.Random`` (linear congruential).

    Only the surface ``WorldOps`` uses is implemented: ``set_seed`` and
    ``next_int(bound)``. Matches the JDK ``Random`` spec so a given seed yields
    the identical jitter sequence to the real Fabric mod.
    """

    _MULT = 0x5DEECE66D
    _ADD = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        # Java: this.seed = (seed ^ 0x5DEECE66D) & ((1L << 48) - 1)
        self._state = (int(seed) ^ self._MULT) & self._MASK

    def _next(self, bits: int) -> int:
        # Java: seed = (seed * 0x5DEECE66D + 0xB) & ((1L << 48) - 1)
        #       return (int)(seed >>> (48 - bits))
        self._state = (self._state * self._MULT + self._ADD) & self._MASK
        return self._state >> (48 - bits)

    def next_int(self, bound: int) -> int:
        # Java Random.nextInt(int bound) for non-power-of-two uses rejection
        # sampling. bound=3 (our only use) is not a power of two.
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:  # power of two
            return (bound * self._next(31)) >> 31
        while True:
            bits = self._next(31)
            val = bits % bound
            # Reject when the draw would bias the distribution (overflow guard).
            if bits - val + (bound - 1) >= 0:
                return val


@dataclass
class SimWorld:
    """World state for one gatherer episode.

    ``log_alive`` is a boolean mask parallel to ``logs`` so that harvesting a log
    flips its bit to ``False`` without deleting rows (indices stay stable).
    """

    agent_pos: np.ndarray = field(default_factory=lambda: SPAWN_POS.copy())  # float (3,)
    logs: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID * GRID, 3), dtype=np.int64)
    )  # int (64, 3) -> (x, y, z)
    log_alive: np.ndarray = field(
        default_factory=lambda: np.zeros(GRID * GRID, dtype=bool)
    )  # bool (64,)
    inventory: dict = field(default_factory=dict)
    tick: int = 0

    def reset(self, seed: int, arena_mode: str = "trees") -> None:
        """Reset to a seeded oak-trunk forest.

        ``arena_mode``:
          - ``"trees"`` (default): the 4x4 16-trunk grid that mirrors WorldOps.java
            byte-for-byte (golden-trace + transfer parity). UNCHANGED.
          - ``"clusters"``: two 8-trunk clusters with a >perception gap between them
            (cluster A around the spawn = visible; cluster B ~22 blocks south =
            beyond the 16-block perception). SIM-ONLY M2 experiment that forces the
            policy to EXPLORE blind from A to discover B. No Java mirror (yet).
        """
        self.agent_pos = SPAWN_POS.copy()
        self.inventory = {}
        self.tick = 0

        # "mixed" (training): alternate trees / randomized-clusters by seed parity so
        # the policy generalizes across BOTH layouts instead of overfitting one geometry.
        if arena_mode == "mixed":
            arena_mode = "clusters" if (int(seed) % 2 == 1) else "trees"
        rng = _JavaRandom(seed)
        if arena_mode == "clusters":
            bases = self._cluster_bases(rng)
        elif arena_mode == "clusters_omni":
            bases = self._cluster_omni_bases(rng)
        else:
            bases = self._tree_grid_bases(rng)
        n = len(bases) * TRUNK_H
        logs = np.zeros((n, 3), dtype=np.int64)
        idx = 0
        for x, z in bases:  # stack TRUNK_H oak_log per trunk base, LOG_Y .. +TRUNK_H-1
            for dy in range(TRUNK_H):
                logs[idx] = (x, LOG_Y + dy, z)
                idx += 1
        self.logs = logs
        self.log_alive = np.ones(n, dtype=bool)

    @staticmethod
    def _cluster_omni_bases(rng: _JavaRandom) -> list[tuple[int, int]]:
        """Non-degenerate clusters: cluster A near spawn (visible), cluster B on a
        ring of radius `gap` in one of 8 UNIFORM compass directions around spawn.

        Unlike ``_cluster_bases`` (whose B was always SOUTH — a constant heading
        cleared it, so it never tested search), here B can be in ANY direction, so a
        fixed heading clears only the ~3/8 of sectors that overlap it. This arena
        REQUIRES omnidirectional search and is the honest testbed for a scout/producer.
        Sim-only. Deterministic (np.cos/np.sin on a seeded sector — no RNG float).
        """
        ax = SPAWN_X + (rng.next_int(7) - 3)   # cluster A center, spawn-ish ±3
        az = SPAWN_Z + (rng.next_int(7) - 3)
        gap = 24 + rng.next_int(7)             # 24..30 (> 16-block perception)
        sector = rng.next_int(8)               # 8 uniform directions
        angle = float(sector) * (float(np.pi) / 4.0)
        bx = SPAWN_X + int(round(gap * float(np.cos(angle))))
        bz = SPAWN_Z + int(round(gap * float(np.sin(angle))))
        # Wider bounds than `clusters` so the full ring fits (omni tests use
        # arena_half ~40 so the agent can roam to B in any direction).
        omin_x, omax_x, omin_z, omax_z = 24, 104, -88, -8
        used: set[tuple[int, int]] = set()
        bases: list[tuple[int, int]] = []
        for cx, cz in ((ax, az), (bx, bz)):
            for row in range(2):
                for col in range(4):
                    x = cx - 7 + 5 * col + (rng.next_int(3) - 1)
                    z = cz - 3 + 6 * row + (rng.next_int(3) - 1)
                    x = max(omin_x, min(omax_x, x))
                    z = max(omin_z, min(omax_z, z))
                    while (x == SPAWN_X and z == SPAWN_Z) or (x, z) in used:
                        x += 1
                    used.add((x, z))
                    bases.append((x, z))
        return bases

    @staticmethod
    def _tree_grid_bases(rng: _JavaRandom) -> list[tuple[int, int]]:
        """16 trunk bases on the 4x4 grid — draw order MUST match WorldOps.java
        byte-for-byte (row outer 0..3, col inner 0..3, x-jitter THEN z-jitter)."""
        used: set[tuple[int, int]] = set()
        bases: list[tuple[int, int]] = []
        for row in range(TREE_GRID):
            for col in range(TREE_GRID):
                base_x = 52 + 7 * col   # 52,59,66,73
                base_z = -61 + 7 * row  # -61,-54,-47,-40
                x = base_x + (rng.next_int(3) - 1)
                z = base_z + (rng.next_int(3) - 1)
                x = max(MIN_X, min(MAX_X, x))
                z = max(MIN_Z, min(MAX_Z, z))
                while (x == SPAWN_X and z == SPAWN_Z) or (x, z) in used:
                    if x < MAX_X:
                        x += 1
                    elif z < MAX_Z:
                        z += 1
                    elif x > MIN_X:
                        x -= 1
                    else:
                        z -= 1
                used.add((x, z))
                bases.append((x, z))
        return bases

    @staticmethod
    def _cluster_bases(rng: _JavaRandom) -> list[tuple[int, int]]:
        """16 trunk bases in two 8-trunk clusters: A near spawn (visible), B a
        randomized ~22-28 blocks away (beyond the 16-block perception) -> the gap
        forces a blind explore hop. Centers are RANDOMIZED per seed so the policy
        can't overfit one geometry. Sim-only M2 experiment (no WorldOps mirror)."""
        # cluster A near spawn (x,z jittered); cluster B a randomized gap away.
        ax = 58 + (rng.next_int(9) - 4)   # spawn-ish x ±4
        az = -48 + (rng.next_int(7) - 3)  # spawn-ish z ±3
        gap = 22 + rng.next_int(7)        # 22..28 blocks (>perception 16)
        # B direction: south, or east/west-south, randomized so it isn't always -z.
        dirs = [(0, -1), (1, -1), (-1, -1)]
        dxc, dzc = dirs[rng.next_int(3)]
        bx = ax + dxc * gap // 2
        bz = az + dzc * gap
        used: set[tuple[int, int]] = set()
        bases: list[tuple[int, int]] = []
        for cx, cz in ((ax, az), (bx, bz)):
            for row in range(2):
                for col in range(4):
                    x = cx - 7 + 5 * col + (rng.next_int(3) - 1)
                    z = cz - 3 + 6 * row + (rng.next_int(3) - 1)
                    # keep inside a generous bound (clusters mode uses arena_half~34)
                    x = max(46, min(94, x))
                    z = max(-86, min(-14, z))
                    while (x == SPAWN_X and z == SPAWN_Z) or (x, z) in used:
                        x += 1
                    used.add((x, z))
                    bases.append((x, z))
        return bases
