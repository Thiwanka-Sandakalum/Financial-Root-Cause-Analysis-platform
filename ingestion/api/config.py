"""Configuration settings for the ingestion API."""
import os
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables."""

    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # LLM Configuration
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_embedding_model: str = os.getenv(
        "GEMINI_EMBEDDING_MODEL", "models/embedding-001"
    )

    # API Configuration
    api_title: str = "RootAlpha Ingestion API"
    api_version: str = "1.0.0"
    api_description: str = "Document ingestion and knowledge graph pipeline"

    # Optional Authentication
    api_key: Optional[str] = os.getenv("API_KEY", None)

    # LlamaParse Configuration
    llama_cloud_api_key: Optional[str] = os.getenv("LLAMA_CLOUD_API_KEY", None)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return Settings()
