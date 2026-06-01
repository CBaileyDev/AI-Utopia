"""Run the AI Utopia GUI backend.

PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.11 -m aiutopia.api
"""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("AIUTOPIA_API_HOST", "127.0.0.1")
    port = int(os.environ.get("AIUTOPIA_API_PORT", "8777"))
    uvicorn.run("aiutopia.api.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
