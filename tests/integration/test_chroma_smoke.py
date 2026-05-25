"""Marker-gated Chroma smoke test. Runs by default; skips if chromadb
is not installed (CI may want to install only [dev], not the full deps)."""
from __future__ import annotations

import pytest

pytest.importorskip("chromadb")

from aiutopia.common.ids import memory_id_for, new_agent_uuid
from aiutopia.memory.client import open_chroma


pytestmark = pytest.mark.integration


def test_chroma_roundtrip(chroma_dir):
    agent_uuid = new_agent_uuid()
    client = open_chroma(chroma_dir)
    coll = client.get_or_create_collection(memory_id_for(agent_uuid))
    coll.add(
        ids=["rec1"],
        documents=["Bjorn found 3 oak logs near the river."],
        metadatas=[{"importance_score": 0.6, "timestamp": 100,
                     "event_type": "harvest",
                     "participants_csv": f",agent-{agent_uuid},"}],
        embeddings=[[0.1] * 384],     # fake BGE-small vector
    )
    got = coll.query(query_embeddings=[[0.1] * 384], n_results=1,
                     include=["documents", "metadatas", "distances"])
    assert got["documents"][0][0] == "Bjorn found 3 oak logs near the river."
    assert got["metadatas"][0][0]["importance_score"] == 0.6
