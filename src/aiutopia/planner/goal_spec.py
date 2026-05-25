"""§3.1 — Tier 2 Goal Spec Adapter.

Frozen pre-trained BGE-small NL embedding (384-d) + structured features
(128-d) → 512-d goal_embedding. Hard role dispatch (string lookup), no
learned routing. BGE model is lazy-loaded on first call to keep cold
imports fast (tests inject a fake)."""
from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from aiutopia.schemas.plan import Subgoal


class BGEEncoder(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


_ROLE_INDEX: dict[str, int] = {
    "gatherer": 0, "builder": 1, "farmer": 2, "defender": 3,
}
_ROLE_TO_POLICY: dict[str, str] = {
    "gatherer": "gatherer_policy",
    "builder":  "builder_policy",
    "farmer":   "farmer_policy",
    "defender": "defender_policy",
}


# Layout of the 128-d structured feature vector (deterministic):
#   [ 0:  4] role one-hot (4 dims; 1.0 at role index)
#   [ 4: 68] inventory_delta normalized (64 dims)
#   [68: 69] timeout_normalized (1 dim) — timeout_ticks / 12000
#   [69: 70] priority normalized   (1 dim) — priority / 10
#   [70:128] reserved flags        (58 dims; zeros for now, used in later milestones)
_STRUCTURED_DIM = 128


def build_structured_features(subgoal: Subgoal) -> np.ndarray:
    out = np.zeros(_STRUCTURED_DIM, dtype=np.float32)
    out[_ROLE_INDEX[subgoal.role]] = 1.0
    # inventory_delta: hash item name → bucket in [4, 68) so it's stable
    for item, qty in subgoal.goal_specification.target_state.inventory_delta.items():
        bucket = 4 + (hash(item) % 64)
        out[bucket] = max(-1.0, min(1.0, float(qty) / 64.0))
    out[68] = subgoal.goal_specification.termination_conditions.timeout_ticks / 12000.0
    out[69] = subgoal.priority / 10.0
    return out


def build_nl_summary(subgoal: Subgoal) -> str:
    """Compose the NL string that gets BGE-encoded.
    Includes role tag so the embedder sees role context."""
    return f"{subgoal.role}: {subgoal.nl_summary}"


class GoalSpecAdapter:
    def __init__(self, bge: BGEEncoder):
        self._bge = bge

    def embed(self, subgoal: Subgoal) -> np.ndarray:
        nl_vec = self._bge.encode(build_nl_summary(subgoal)).astype(np.float32)
        if nl_vec.shape != (384,):
            raise ValueError(f"BGE encoder returned shape {nl_vec.shape}, expected (384,)")
        struct = build_structured_features(subgoal)
        return np.concatenate([nl_vec, struct]).astype(np.float32)

    def policy_name_for(self, subgoal: Any) -> str:
        # `subgoal.role` is sufficient — tolerate ducktype for testability.
        return _ROLE_TO_POLICY[subgoal.role]


def load_bge_small() -> BGEEncoder:
    """Lazy loader for `BAAI/bge-small-en-v1.5`. Call once at process start;
    NEVER from inside a tight loop. Heavy import."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

    class _Wrapper:
        def encode(self, text: str) -> np.ndarray:
            return model.encode(text, normalize_embeddings=True)

    return _Wrapper()
