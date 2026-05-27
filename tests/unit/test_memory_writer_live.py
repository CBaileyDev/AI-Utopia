import pytest

pytest.importorskip("chromadb")

from aiutopia.common.ids import new_agent_uuid
from aiutopia.memory.client import open_chroma
from aiutopia.memory.writer import EpisodicMemoryWriter, EpisodicRecord

pytestmark = pytest.mark.integration


def test_high_importance_writes_immediately_to_chroma(chroma_dir, tmp_path):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = new_agent_uuid()
    rec = EpisodicRecord(
        agent_uuid=uuid,
        timestamp=42,
        event_type="chopped",
        participants=[],
        importance_score=0.85,  # > 0.7 → immediate
        summary="agent chopped an oak_log",
        embedding=[0.1] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "high"

    coll = client.get_collection(f"mem_{uuid}")
    out = coll.get(limit=10)
    assert len(out["ids"]) == 1
    assert "oak_log" in out["documents"][0]
    assert out["metadatas"][0]["importance_score"] == 0.85


def test_medium_importance_buffered_then_flushed(chroma_dir):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    uuid = new_agent_uuid()
    rec = EpisodicRecord(
        agent_uuid=uuid,
        timestamp=1,
        event_type="step",
        participants=[],
        importance_score=0.45,
        summary="agent walked",
        embedding=[0.0] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "medium"
    # Not yet written to Chroma — verify by querying empty
    coll = client.get_or_create_collection(f"mem_{uuid}")
    assert len(coll.get()["ids"]) == 0
    # Flush
    flushed = writer.flush()
    assert flushed >= 1
    # Now present
    assert len(coll.get()["ids"]) >= 1


def test_low_importance_skipped(chroma_dir):
    client = open_chroma(chroma_dir)
    writer = EpisodicMemoryWriter(chroma_client=client)
    rec = EpisodicRecord(
        agent_uuid=new_agent_uuid(), timestamp=1, event_type="noise",
        participants=[], importance_score=0.1, summary="x", embedding=[0.0] * 384,
    )
    bucket = writer.maybe_write(rec)
    assert bucket == "skipped"
