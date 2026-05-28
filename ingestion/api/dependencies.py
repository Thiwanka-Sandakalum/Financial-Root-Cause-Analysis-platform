"""FastAPI dependencies (Dependency Injection)."""
from functools import lru_cache
from typing import AsyncGenerator

from fastapi import Depends
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from neo4j import AsyncDriver, AsyncGraphDatabase

from ingestion.api.config import Settings, get_settings


@lru_cache
def get_cached_settings() -> Settings:
    """Return a cached Settings instance (parsed once at startup)."""
    return get_settings()


async def get_neo4j_driver(
    settings: Settings = Depends(get_cached_settings),
) -> AsyncGenerator[AsyncDriver, None]:
    """Yield an async Neo4j driver; closed after the request/task finishes."""
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        await driver.verify_connectivity()
        yield driver
    finally:
        await driver.close()


async def get_llm(
    settings: Settings = Depends(get_cached_settings),
) -> ChatGoogleGenerativeAI:
    """Return a Gemini LLM client."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=0.0,
    )


async def get_embedder(
    settings: Settings = Depends(get_cached_settings),
) -> GoogleGenerativeAIEmbeddings:
    """Return a Google Generative AI embeddings client."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
    )

