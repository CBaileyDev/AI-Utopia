"""SimWorld: fast headless world state for the gatherer M1B task.

Mirrors ``WorldOps.resetEpisode`` (fabric_mod/.../bridge/WorldOps.java) exactly:
16 vertical BARE oak trunks (4 logs each, Y=66..69 = 64 logs total) on a 4x4
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
LOG_Y = 66
GRID = 8  # legacy: 8x8 == 64 (kept so logs/log_alive default factories size to 64)

# N21 Inc2: the arena is 16 vertical BARE oak trunks (4 logs each, Y=66..69),
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

    def reset(self, seed: int) -> None:
        """Reset to the seeded 16-trunk forest, mirroring ``resetEpisode`` exactly."""
        self.agent_pos = SPAWN_POS.copy()
        self.inventory = {}
        self.tick = 0

        rng = _JavaRandom(seed)
        logs = np.zeros((TREES * TRUNK_H, 3), dtype=np.int64)
        used: set[tuple[int, int]] = set()
        idx = 0
        # Iteration + draw order MUST match WorldOps.java byte-for-byte: row outer
        # (0..3), col inner (0..3); per trunk draw jitter for x THEN z (two
        # next_int(3) calls), then stack TRUNK_H logs from LOG_Y up. NO height
        # draw (fixed 4) -> the RNG draw pattern is the same 2-per-cell surface as
        # the old flat grid, keeping parity simple.
        for row in range(TREE_GRID):
            for col in range(TREE_GRID):
                base_x = 52 + 7 * col   # 52,59,66,73
                base_z = -61 + 7 * row  # -61,-54,-47,-40
                x = base_x + (rng.next_int(3) - 1)
                z = base_z + (rng.next_int(3) - 1)
                # Clamp to arena (defensive; grid+jitter never exceeds).
                x = max(MIN_X, min(MAX_X, x))
                z = max(MIN_Z, min(MAX_Z, z))
                # Resolve collisions with the spawn tile or an already-placed trunk
                # by nudging deterministically through neighbor offsets.
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
                for dy in range(TRUNK_H):  # stack the trunk: LOG_Y .. LOG_Y+TRUNK_H-1
                    logs[idx] = (x, LOG_Y + dy, z)
                    idx += 1

        self.logs = logs
        self.log_alive = np.ones(TREES * TRUNK_H, dtype=bool)
