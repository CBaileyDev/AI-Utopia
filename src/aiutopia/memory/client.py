"""§5.6 — Chroma client wrapper. Local persistent store under chroma_dir."""
from __future__ import annotations

from pathlib import Path

import chromadb


def open_chroma(chroma_dir: Path) -> chromadb.ClientAPI:
    """Open a persistent local Chroma client rooted at chroma_dir."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))
