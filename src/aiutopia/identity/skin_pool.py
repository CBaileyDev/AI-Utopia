"""Agent name + skin selection per §3.5 conventions.

- 12 names per role; on exhaustion fall back to "<first> <procedural-surname>".
- Skin selection is deterministic per agent_uuid (NOT per agent_name) so two
  agents that share a name (across lives) look different.
- Procedural surnames combine fixed roots and suffixes; never numeric.
"""
from __future__ import annotations

import hashlib
import random


_SURNAME_ROOTS = (
    "Iron", "Stone", "Oak", "Ash", "Pine", "Frost", "Storm", "River",
    "Hawk", "Wolf", "Bear", "Raven", "Fox", "Shadow", "Moon", "Sun",
)
_SURNAME_SUFFIXES = (
    "wood", "stone", "field", "hold", "bane", "ward", "blade",
    "thorn", "song", "claw", "fall", "gate", "vale", "mark",
)


def procedural_surname(seed: int) -> str:
    rng = random.Random(seed)
    return rng.choice(_SURNAME_ROOTS) + rng.choice(_SURNAME_SUFFIXES)


def pick_name(pool: list[str], used: set[str], seed: int | None = None) -> str:
    """Return an unused name from `pool`, or a procedural fallback when
    every pool name is in `used`.

    Determinism: when `seed` is given, fallback is reproducible. When None,
    a fresh random is used (production succession; succession_seed should
    be derived from current tick for replay).
    """
    available = [n for n in pool if n not in used]
    rng = random.Random(seed)
    if available:
        return rng.choice(available)
    first = rng.choice(pool)
    return f"{first} {procedural_surname(seed=rng.randrange(1 << 32))}"


def deterministic_skin_for_uuid(agent_uuid: str, skins: list[str]) -> str:
    """Deterministic pick from `skins` indexed by hash(agent_uuid)."""
    if not skins:
        raise ValueError("skin pool empty")
    h = hashlib.sha256(agent_uuid.encode()).digest()
    idx = int.from_bytes(h[:8], "big") % len(skins)
    return skins[idx]
