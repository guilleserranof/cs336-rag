"""Tests for application settings."""

from pathlib import Path

from cs336_rag.config import Settings


def make_settings(**overrides: object) -> Settings:
    """Build Settings without reading the local .env file."""
    defaults: dict[str, object] = {"openai_key": "test-key"}
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)  # type: ignore[call-arg]


def test_defaults() -> None:
    settings = make_settings()
    assert settings.openai_key == "test-key"
    assert settings.llm_base_url == "https://api.nan.builders/v1"
    assert settings.chat_model == "qwen3.6"
    assert settings.embedding_model == "qwen3-embedding"
    assert settings.embedding_dim == 1024
    assert settings.rerank_model == "rerank"
    assert settings.data_dir == Path("data")


def test_openai_key_is_optional() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_key is None


def test_env_overrides(monkeypatch: object) -> None:
    import pytest

    mp = monkeypatch
    assert isinstance(mp, pytest.MonkeyPatch)
    mp.setenv("OPENAI_KEY", "from-env")
    mp.setenv("CHAT_MODEL", "gemma4")
    mp.setenv("DB_PORT", "5433")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openai_key == "from-env"
    assert settings.chat_model == "gemma4"
    assert settings.db_port == 5433


def test_db_dsn() -> None:
    settings = make_settings(
        db_host="db", db_port=5432, db_name="cs336_rag", db_user="u", db_password="p"
    )
    assert settings.db_dsn == "postgresql://u:p@db:5432/cs336_rag"
