from aiutopia.memory.writer import (
    BATCH_FLUSH_MAX_RECORDS,
    IMPORTANCE_WEIGHTS,
    EpisodicMemoryWriter,
    EpisodicRecord,
    importance_score,
)


class _FakeCollection:
    def __init__(self):
        self.ids: list[str] = []

    def add(self, *, ids, documents, metadatas, embeddings):
        self.ids.extend(ids)


class _FakeChroma:
    """Minimal stand-in for chromadb.ClientAPI for unit tests (one coll per name)."""

    def __init__(self):
        self.collections: dict[str, _FakeCollection] = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, _FakeCollection())


def _medium(uuid: str, ts: int) -> EpisodicRecord:
    return EpisodicRecord(
        agent_uuid=uuid,
        timestamp=ts,
        event_type="step",
        participants=[],
        importance_score=0.45,  # MEDIUM → buffered
        summary="x",
        embedding=[0.0] * 384,
    )


def test_importance_weights_sum_to_one() -> None:
    assert abs(sum(IMPORTANCE_WEIGHTS.values()) - 1.0) < 1e-9


def test_importance_score_all_zero_inputs_is_zero() -> None:
    s = importance_score(
        abs_reward_norm=0.0,
        novel_state=0.0,
        comm_norm=0.0,
        player_proximity=0.0,
        threat_level=0.0,
        planner_event=0.0,
    )
    assert s == 0.0


def test_importance_score_all_one_inputs_is_one() -> None:
    s = importance_score(
        abs_reward_norm=1.0,
        novel_state=1.0,
        comm_norm=1.0,
        player_proximity=1.0,
        threat_level=1.0,
        planner_event=1.0,
    )
    assert abs(s - 1.0) < 1e-9


def test_importance_score_weighted_combination_matches_spec() -> None:
    # Only abs_reward_norm = 1, others 0 → should equal 0.30 weight
    s = importance_score(
        abs_reward_norm=1.0,
        novel_state=0.0,
        comm_norm=0.0,
        player_proximity=0.0,
        threat_level=0.0,
        planner_event=0.0,
    )
    assert abs(s - IMPORTANCE_WEIGHTS["abs_reward"]) < 1e-9


def test_buffer_auto_flushes_at_batch_cap() -> None:
    """The MEDIUM buffer must be BOUNDED: it auto-flushes at BATCH_FLUSH_MAX_RECORDS
    so it can't grow without limit when the caller never calls flush() (the leak)."""
    client = _FakeChroma()
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = "01J0000000000000000000000A"
    for i in range(BATCH_FLUSH_MAX_RECORDS - 1):
        assert writer.maybe_write(_medium(uuid, i)) == "medium"
    # One below the cap: still buffered, nothing written.
    assert writer._buffered == BATCH_FLUSH_MAX_RECORDS - 1
    assert sum(len(c.ids) for c in client.collections.values()) == 0
    # The record that hits the cap triggers an auto-flush.
    writer.maybe_write(_medium(uuid, BATCH_FLUSH_MAX_RECORDS - 1))
    assert writer._buffered == 0
    assert sum(len(c.ids) for c in client.collections.values()) == BATCH_FLUSH_MAX_RECORDS


def test_chroma_ids_unique_across_flushes_same_timestamp() -> None:
    """Two records for the same (agent, timestamp) in DIFFERENT flushes must get
    DISTINCT ids — the old per-call enumerate index collided and Chroma's add()
    silently overwrote the earlier memory."""
    client = _FakeChroma()
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = "01J0000000000000000000000B"
    writer.maybe_write(_medium(uuid, 7))
    writer.flush()
    writer.maybe_write(_medium(uuid, 7))  # same timestamp, separate batch
    writer.flush()
    coll = client.collections[f"mem_{uuid}"]
    assert len(coll.ids) == 2
    assert len(set(coll.ids)) == 2, f"ids collided across flushes: {coll.ids}"
