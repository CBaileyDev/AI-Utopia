"""Shared, import-light embeds + the gatherer goal stub.

Lifted out of `env/wrapper.py` (and the inline Subgoal in its `__init__`) so
the fast-sim package (`aiutopia.sim.*`) can reuse the *exact* constant embeds
and the success predicate WITHOUT transitively importing the heavy deps the
wrapper pulls in (chromadb / py4j / sentence_transformers / torch).

This module imports ONLY:
  - stdlib (`hashlib`)
  - `numpy`
  - the light pydantic schema (`aiutopia.schemas.plan`)
  - `aiutopia.planner.goal_spec.build_structured_features` — itself import-light
    (zlib/typing/numpy/schemas only; the heavy BGE import is lazy, inside
    `load_bge_small`, which we never call here).

Parity note (read before changing the goal stub): the wrapper builds its
`goal_embedding` by running the *same* `gatherer_stub_subgoal()` Subgoal
through `GoalSpecAdapter.embed`, which prepends a 384-d BGE NL vector (real
when sentence-transformers is installed, zeros via `_ZeroBGE` otherwise) to
the 128-d structured features. The sim is import-light and cannot run BGE, so
`gatherer_goal_embedding_stub()` here is the BGE-ABSENT case byte-for-byte:
zeros(384) + structured(128). When BGE is present in the wrapper's process the
two diverge on the first 384 dims; that divergence is intentional and out of
scope for Phase A (goal_embedding is not a golden-trace DYNAMIC field).
"""

from __future__ import annotations

import hashlib

import numpy as np

from aiutopia.planner.goal_spec import build_structured_features
from aiutopia.schemas.plan import (
    Constraints,
    GoalSpecification,
    Subgoal,
    TargetState,
    TerminationConditions,
)

# Must match spaces.py (kept here so wrapper + sim share one source).
_AGENT_UUID_EMBED_DIM = 384
_GOAL_NL_DIM = 384  # BGE part of the 512-d goal_embedding (zeros in the stub)


def _agent_uuid_embed(uuid_str: str) -> np.ndarray:
    """Deterministic 384-d float32 embed from agent UUID string.
    SHA-256 → 32 bytes → 384 floats (tile to fill) → normalized [-1, 1].
    Stable across processes (M0 hash() carry-forward + R10)."""
    digest = hashlib.sha256(uuid_str.encode("utf-8")).digest()  # 32 bytes
    # Tile 32 bytes -> 384 bytes (32 x 12 = 384)
    tiled = (digest * 12)[:_AGENT_UUID_EMBED_DIM]
    arr = np.frombuffer(tiled, dtype=np.uint8).astype(np.float32)
    return (arr / 127.5) - 1.0  # → [-1, 1]


def _goal_success(goal_spec: GoalSpecification, inventory: dict[str, int]) -> bool:
    """Phase-0 fix #5 — pure success predicate for SUCCESS-TERMINATION.

    Evaluates the GoalSpec's success condition against a fully-reconstructed
    `{item_name: count}` inventory dict (built at the call site via
    reward._inventory_from_obs, the same function the reward path uses, so
    success and reward never disagree about the bag's contents).

    Currently implements the single `inventory_meets_delta` criterion the M1B
    gatherer goal encodes (collect 64 oak_log). The check is ABSOLUTE: M1B's
    reset_episode clears the inventory before every episode, so "delta from
    empty" == "current count >= target". Semantics are `>=` per item, so extra
    off-task items (e.g. cobblestone) don't disqualify a met target.

    Returns False — never spuriously True — when:
      - the goal uses a criterion this helper does not implement (spatial /
        blueprint / threat target); we are honest about the one we cover, and
      - the inventory_delta is empty (guards the `all([]) == True` trap so an
        empty target never reads as an instant win).
    """
    tc = goal_spec.termination_conditions
    if "inventory_meets_delta" not in tc.success_criteria:
        return False
    delta = goal_spec.target_state.inventory_delta
    return bool(delta) and all(inventory.get(item, 0) >= qty for item, qty in delta.items())


def gatherer_stub_subgoal() -> Subgoal:
    """The hardcoded M1B "collect 64 oak_log" Subgoal.

    Single source of truth for both the wrapper (which BGE-embeds it into
    `goal_embedding`) and the sim (which uses its goal_specification for the
    `_goal_success` termination check). Plan B replaces this with planner-
    emitted Subgoals routed via aiutopia.planner.event_queue.
    """
    return Subgoal(
        role="gatherer",
        priority=5,
        goal_specification=GoalSpecification(
            target_state=TargetState(inventory_delta={"oak_log": 64}),
            termination_conditions=TerminationConditions(
                success_criteria=["inventory_meets_delta"],
                timeout_ticks=6000,
            ),
        ),
        constraints=Constraints(),
        nl_summary="collect 64 oak_log",
    )


def gatherer_goal_embedding_stub() -> np.ndarray:
    """The 512-d `goal_embedding` for the gatherer stub, BGE-ABSENT.

    Equals the wrapper's `_ZeroBGE` fallback byte-for-byte: 384 zeros (NL part)
    concatenated with the 128-d structured features of `gatherer_stub_subgoal()`.
    Import-light — never touches sentence-transformers. See the module docstring
    for the BGE-present divergence note.
    """
    nl = np.zeros(_GOAL_NL_DIM, dtype=np.float32)
    struct = build_structured_features(gatherer_stub_subgoal())
    return np.concatenate([nl, struct]).astype(np.float32)
