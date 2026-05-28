"""Run the FastAPI ingestion API server."""
import logging
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


def _load_env() -> None:
    """Load the API env file regardless of the current working directory."""
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path, override=False)

if __name__ == "__main__":
    _load_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(
        "ingestion.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
