from functools import lru_cache
from typing import Any

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    neo4j_uri: str = Field(alias="NEO4J_URI")
    neo4j_username: str = Field(alias="NEO4J_USERNAME")
    neo4j_password: str = Field(alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")

    gemini_use_vertexai: bool = Field(default=False, alias="GEMINI_USE_VERTEXAI")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    google_cloud_project: str | None = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field(default="us-central1", alias="GOOGLE_CLOUD_LOCATION")
    gemini_fast_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_FAST_MODEL")
    gemini_strong_model: str = Field(default="gemini-2.5-pro", alias="GEMINI_STRONG_MODEL")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL"
    )

    request_timeout_seconds: int = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    max_retry_attempts: int = Field(default=3, alias="MAX_RETRY_ATTEMPTS")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")
    ingestion_chunk_size: int = Field(default=1200, alias="INGESTION_CHUNK_SIZE")
    ingestion_chunk_overlap: int = Field(default=200, alias="INGESTION_CHUNK_OVERLAP")
    ingestion_min_chunk_length: int = Field(default=100, alias="INGESTION_MIN_CHUNK_LENGTH")
    ingestion_write_batch_size: int = Field(default=100, alias="INGESTION_WRITE_BATCH_SIZE")
    ingestion_embedding_batch_size: int = Field(default=32, alias="INGESTION_EMBEDDING_BATCH_SIZE")
    ingestion_extraction_batch_size: int = Field(
        default=16,
        alias="INGESTION_EXTRACTION_BATCH_SIZE",
    )
    ingestion_extraction_max_concurrency: int = Field(
        default=4,
        alias="INGESTION_EXTRACTION_MAX_CONCURRENCY",
    )
    extraction_confidence_threshold: float = Field(
        default=0.7,
        alias="EXTRACTION_CONFIDENCE_THRESHOLD",
    )
    similarity_threshold: float = Field(default=0.8, alias="SIMILARITY_THRESHOLD")
    supported_upload_types: str = Field(default=".pdf,.txt", alias="SUPPORTED_UPLOAD_TYPES")

    query_retrieval_top_k: int = Field(default=5, alias="QUERY_RETRIEVAL_TOP_K")
    query_evidence_top_k: int = Field(default=10, alias="QUERY_EVIDENCE_TOP_K")
    query_synthesis_top_k: int = Field(default=8, alias="QUERY_SYNTHESIS_TOP_K")
    query_graph_expansion_hops: int = Field(default=2, alias="QUERY_GRAPH_EXPANSION_HOPS")
    query_similarity_threshold: float = Field(
        default=0.75,
        alias="QUERY_SIMILARITY_THRESHOLD",
    )
    query_ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "relevance": 0.6,
            "temporal": 0.2,
            "confidence": 0.2,
        },
        alias="QUERY_RANKING_WEIGHTS",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_value = value.upper()
        if upper_value not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}")
        return upper_value

    @field_validator(
        "request_timeout_seconds",
        "max_retry_attempts",
        "embedding_dimensions",
        "ingestion_extraction_batch_size",
        "ingestion_extraction_max_concurrency",
        "query_retrieval_top_k",
        "query_evidence_top_k",
        "query_synthesis_top_k",
        "query_graph_expansion_hops",
    )
    @classmethod
    def validate_positive_values(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Value must be greater than 0")
        return value

    @field_validator(
        "extraction_confidence_threshold",
        "similarity_threshold",
        "query_similarity_threshold",
    )
    @classmethod
    def validate_probability_values(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("Value must be between 0 and 1")
        return value

    @field_validator("supported_upload_types")
    @classmethod
    def validate_supported_upload_types(cls, value: str) -> str:
        normalized_types = [item.strip().lower() for item in value.split(",") if item.strip()]
        if not normalized_types:
            raise ValueError("SUPPORTED_UPLOAD_TYPES must contain at least one file extension")
        if any(not item.startswith(".") for item in normalized_types):
            raise ValueError("SUPPORTED_UPLOAD_TYPES values must start with '.'")
        return ",".join(normalized_types)

    @field_validator("query_ranking_weights")
    @classmethod
    def validate_query_ranking_weights(cls, value: Any) -> dict[str, float]:
        required_keys = {"relevance", "temporal", "confidence"}
        if not isinstance(value, dict):
            raise ValueError("QUERY_RANKING_WEIGHTS must be a JSON object")

        if set(value.keys()) != required_keys:
            raise ValueError(
                "QUERY_RANKING_WEIGHTS must include exactly: relevance, temporal, confidence"
            )

        normalized = {key: float(raw_value) for key, raw_value in value.items()}
        for key, raw_value in normalized.items():
            if raw_value < 0 or raw_value > 1:
                raise ValueError(f"QUERY_RANKING_WEIGHTS[{key}] must be between 0 and 1")

        total = sum(normalized.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("QUERY_RANKING_WEIGHTS must sum to 1.0")

        return normalized

    @model_validator(mode="after")
    def validate_google_auth_mode(self) -> "Settings":
        if self.gemini_use_vertexai:
            if not self.google_cloud_project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT is required when GEMINI_USE_VERTEXAI=true"
                )
        elif not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when GEMINI_USE_VERTEXAI=false")
        if self.ingestion_chunk_overlap >= self.ingestion_chunk_size:
            raise ValueError("INGESTION_CHUNK_OVERLAP must be smaller than INGESTION_CHUNK_SIZE")
        if self.ingestion_min_chunk_length > self.ingestion_chunk_size:
            raise ValueError(
                "INGESTION_MIN_CHUNK_LENGTH must be less than or equal to INGESTION_CHUNK_SIZE"
            )
        return self

    @property
    def supported_upload_extensions(self) -> tuple[str, ...]:
        return tuple(self.supported_upload_types.split(","))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        raise RuntimeError(f"Configuration validation failed: {exc}") from exc
