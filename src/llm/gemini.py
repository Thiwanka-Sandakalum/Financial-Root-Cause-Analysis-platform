import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import Settings


def _configure_google_backend(settings: Settings) -> None:
    if settings.gemini_use_vertexai:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project or ""
        os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location


def _chat_kwargs(settings: Settings) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "temperature": 0,
    }
    if not settings.gemini_use_vertexai and settings.gemini_api_key:
        kwargs["google_api_key"] = settings.gemini_api_key
    return kwargs


def build_fast_model(settings: Settings) -> ChatGoogleGenerativeAI:
    _configure_google_backend(settings)
    return ChatGoogleGenerativeAI(
        model=settings.gemini_fast_model,
        **_chat_kwargs(settings),
    )


def build_strong_model(settings: Settings) -> ChatGoogleGenerativeAI:
    _configure_google_backend(settings)
    kwargs = _chat_kwargs(settings)
    kwargs["thinking_budget"] = 512
    return ChatGoogleGenerativeAI(
        model=settings.gemini_strong_model,
        **kwargs,
    )


def build_embedding_model(settings: Settings) -> GoogleGenerativeAIEmbeddings:
    _configure_google_backend(settings)
    kwargs: dict[str, str] = {}
    if not settings.gemini_use_vertexai and settings.gemini_api_key:
        kwargs["google_api_key"] = settings.gemini_api_key

    return GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        output_dimensionality=settings.embedding_dimensions,
        **kwargs,
    )
