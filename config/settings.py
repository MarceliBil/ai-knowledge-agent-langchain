from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _get_env(name: str, *, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, default)
    if required and (value is None or value.strip() == ""):
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Add it to your .env file or set it in your system environment."
        )
    return value or ""


def _get_env_int(name: str, *, default: int, required: bool = False) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        if required:
            raise RuntimeError(
                f"Missing required environment variable: {name}. "
                "Add it to your .env file or set it in your system environment."
            )
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise RuntimeError(
            f"{name} must be an integer, got: {raw!r}"
        ) from e


def _get_env_float(name: str, *, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise RuntimeError(f"{name} must be a number, got: {raw!r}") from e


@dataclass(frozen=True)
class Settings:
    azure_search_endpoint: str
    azure_search_key: str
    azure_search_index: str

    azure_storage_connection_string: str
    azure_storage_container: str

    openai_api_key: str
    anthropic_api_key: str | None
    embedding_model: str
    llm_model: str

    retrieval_k: int
    retrieval_search_type: str
    retrieval_score_threshold: float | None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        azure_search_endpoint=_get_env("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=_get_env("AZURE_SEARCH_KEY"),
        azure_search_index=_get_env("AZURE_SEARCH_INDEX"),
        azure_storage_connection_string=_get_env(
            "AZURE_STORAGE_CONNECTION_STRING"),
        azure_storage_container=_get_env("AZURE_STORAGE_CONTAINER"),
        openai_api_key=_get_env("OPENAI_API_KEY"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        embedding_model=os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        llm_model=os.environ.get("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        retrieval_k=_get_env_int("RAG_RETRIEVAL_K", default=6),
        retrieval_search_type=os.environ.get("RAG_SEARCH_TYPE", "hybrid"),
        retrieval_score_threshold=_get_env_float(
            "RAG_SCORE_THRESHOLD", default=None),
    )
