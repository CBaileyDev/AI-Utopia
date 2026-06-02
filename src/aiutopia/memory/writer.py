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
    "abs_reward": 0.30,
    "novel_state": 0.15,
    "comm_norm": 0.10,
    "player_proximity": 0.15,
    "threat_level": 0.15,
    "planner_event": 0.15,
}

# Tiered thresholds (§4.9)
HIGH_IMPORTANCE_THRESHOLD = 0.70
MEDIUM_IMPORTANCE_THRESHOLD = 0.30

# Batch flush limits (§4.9)
BATCH_FLUSH_EVERY_TICKS = 200
BATCH_FLUSH_MAX_RECORDS = 50


def importance_score(
    *,
    abs_reward_norm: float,
    novel_state: float,
    comm_norm: float,
    player_proximity: float,
    threat_level: float,
    planner_event: float,
) -> float:
    """Weighted sum of clamped [0, 1] inputs → score in [0, 1]."""
    parts = {
        "abs_reward": abs_reward_norm,
        "novel_state": novel_state,
        "comm_norm": comm_norm,
        "player_proximity": player_proximity,
        "threat_level": threat_level,
        "planner_event": planner_event,
    }
    return float(sum(IMPORTANCE_WEIGHTS[k] * max(0.0, min(1.0, v)) for k, v in parts.items()))


@dataclass
class EpisodicRecord:
    agent_uuid: str
    timestamp: int
    event_type: str
    participants: list[str]
    importance_score: float
    summary: str
    embedding: list[float] | None = None


@dataclass
class EpisodicMemoryWriter:
    """Importance-weighted writer.

    M1-Pipeline change vs M0: now optionally takes a Chroma client and writes
    real records. When `chroma_client=None` falls back to M0 buffer-only
    counting (useful for unit tests that don't want Chroma overhead)."""

    chroma_client: Any = None  # chromadb.ClientAPI or None
    high_count: int = 0
    medium_count: int = 0
    skipped_count: int = 0
    _buffer: dict[str, list[EpisodicRecord]] = field(default_factory=lambda: defaultdict(list))
    # Monotonic per-writer sequence for Chroma ids (collision-safe across flushes).
    _write_seq: int = 0
    # Total records currently buffered (kept in sync so the auto-flush check is O(1),
    # not O(agents) per write).
    _buffered: int = 0

    def maybe_write(self, record: EpisodicRecord) -> str:
        if record.importance_score >= HIGH_IMPORTANCE_THRESHOLD:
            self.high_count += 1
            self._write_to_chroma([record])
            return "high"
        if record.importance_score >= MEDIUM_IMPORTANCE_THRESHOLD:
            self.medium_count += 1
            self._buffer[record.agent_uuid].append(record)
            self._buffered += 1
            # Auto-flush at the documented batch cap so the buffer is BOUNDED even
            # if the caller never calls flush() (previously BATCH_FLUSH_MAX_RECORDS
            # was defined but unused — the buffer grew without limit across a long
            # episode, an unbounded-memory leak, and §4.9's batch-write was never
            # actually batched). Also amortizes Chroma add() over MAX_RECORDS.
            if self._buffered >= BATCH_FLUSH_MAX_RECORDS:
                self.flush()
            return "medium"
        self.skipped_count += 1
        return "skipped"

    def flush(self) -> int:
        """Flush all buffered MEDIUM records to Chroma. Returns count written."""
        total = 0
        for _agent_uuid, records in list(self._buffer.items()):
            if records:
                self._write_to_chroma(records)
                total += len(records)
        self._buffer.clear()
        self._buffered = 0
        return total

    # ─────────────────────────────────────────────────────────────
    def _write_to_chroma(self, records: list[EpisodicRecord]) -> None:
        if self.chroma_client is None:
            return  # M0-compatible no-op mode
        from aiutopia.common.ids import memory_id_for

        # Group by agent_uuid; one collection per agent
        by_agent: dict[str, list[EpisodicRecord]] = defaultdict(list)
        for r in records:
            by_agent[r.agent_uuid].append(r)
        for agent_uuid, recs in by_agent.items():
            coll = self.chroma_client.get_or_create_collection(memory_id_for(agent_uuid))
            # Ids use a per-WRITER monotonic sequence, not a per-CALL enumerate index.
            # The old `…-{i}` collided across flushes: two records for the same agent
            # at the same timestamp in different batches both got `…-0`, and Chroma's
            # add() silently OVERWRITES a duplicate id → the earlier memory was lost.
            ids = []
            for r in recs:
                ids.append(f"{r.agent_uuid}-{r.timestamp}-{self._write_seq}")
                self._write_seq += 1
            docs = [r.summary for r in recs]
            metas = [
                {
                    "timestamp": r.timestamp,
                    "event_type": r.event_type,
                    "importance_score": r.importance_score,
                    "participants_csv": "," + ",".join(r.participants) + ","
                    if r.participants
                    else ",",
                }
                for r in recs
            ]
            embeds = [r.embedding if r.embedding is not None else [0.0] * 384 for r in recs]
            coll.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)
