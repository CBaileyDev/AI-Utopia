"""Launch the AI Utopia GUI backend (FastAPI + uvicorn) on 127.0.0.1:8777.

Usage:
    PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 scripts/run_api.py
    # or equivalently:
    PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 -m aiutopia.api

Env overrides: AIUTOPIA_API_HOST, AIUTOPIA_API_PORT.
"""

from __future__ import annotations

from aiutopia.api.__main__ import main

if __name__ == "__main__":
    main()
