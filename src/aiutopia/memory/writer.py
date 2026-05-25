"""§4.9 — Episodic memory write path with importance scoring.

In M0 only the scoring + batch-buffer logic is implemented (no LLM summary
generation; that's M5). The writer can persist to Chroma if a client is
passed; otherwise it batches in memory for tests."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

# v1 weights — starting guesses. Exploitation hunt at end of M3 may revise.
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "abs_reward":      0.30,
    "novel_state":     0.15,
    "comm_norm":       0.10,
    "player_proximity":0.15,
    "threat_level":    0.15,
    "planner_event":   0.15,
}

# Tiered thresholds (§4.9)
HIGH_IMPORTANCE_THRESHOLD   = 0.70
MEDIUM_IMPORTANCE_THRESHOLD = 0.30

# Batch flush limits (§4.9)
BATCH_FLUSH_EVERY_TICKS   = 200
BATCH_FLUSH_MAX_RECORDS   = 50


def importance_score(*,
                     abs_reward_norm: float,
                     novel_state: float,
                     comm_norm: float,
                     player_proximity: float,
                     threat_level: float,
                     planner_event: float) -> float:
    """Weighted sum of clamped [0, 1] inputs → score in [0, 1]."""
    parts = {
        "abs_reward":       abs_reward_norm,
        "novel_state":      novel_state,
        "comm_norm":        comm_norm,
        "player_proximity": player_proximity,
        "threat_level":     threat_level,
        "planner_event":    planner_event,
    }
    return float(sum(IMPORTANCE_WEIGHTS[k] * max(0.0, min(1.0, v))
                     for k, v in parts.items()))


@dataclass
class EpisodicRecord:
    agent_uuid:       str
    timestamp:        int
    event_type:       str
    participants:     list[str]
    importance_score: float
    summary:          str
    embedding:        list[float] | None = None


@dataclass
class EpisodicMemoryWriter:
    """Buffers MEDIUM-importance records, immediate-writes HIGH ones.
    Real Chroma writes are wired in M5 alongside summary-generation LLM
    calls; in M0 this only buffers + counts so smoke tests are testable."""
    high_count:   int = 0
    medium_count: int = 0
    skipped_count: int = 0
    _buffer: dict[str, list[EpisodicRecord]] = field(
        default_factory=lambda: defaultdict(list))

    def maybe_write(self, record: EpisodicRecord) -> str:
        if record.importance_score >= HIGH_IMPORTANCE_THRESHOLD:
            self.high_count += 1
            return "high"
        if record.importance_score >= MEDIUM_IMPORTANCE_THRESHOLD:
            self.medium_count += 1
            self._buffer[record.agent_uuid].append(record)
            return "medium"
        self.skipped_count += 1
        return "skipped"

    def flush(self) -> int:
        flushed = sum(len(v) for v in self._buffer.values())
        self._buffer.clear()
        return flushed
