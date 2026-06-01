"""AI Utopia GUI <-> Python backend (FastAPI).

Bridges the Tauri desktop GUI to the existing Python systems (identity.db,
FabricBridge, RLlib training runs, reward config). Heavy deps (chromadb, py4j,
ray, torch) are lazy-imported INSIDE route handlers so the server starts in
<1s and a dead Minecraft server only fails the agent routes — /api/health and
/api/training/* work with no Minecraft and no heavy deps loaded.

Launch:  PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 -m aiutopia.api
"""

from __future__ import annotations

from aiutopia.api.app import create_app

__all__ = ["create_app"]
