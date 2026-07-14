"""Application settings, loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CS336_PLAYLIST_ID = "PLoROMvodv4rMqXOcazWaTUHhq-yembLCV"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI-compatible API (NaN)
    openai_key: str
    llm_base_url: str = "https://api.nan.builders/v1"
    chat_model: str = "qwen3.6"
    judge_model: str = "deepseek-v4-flash"
    embedding_model: str = "qwen3-embedding"
    embedding_dim: int = 1024
    rerank_model: str = "rerank"
    whisper_model: str = "whisper"

    # Postgres
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "cs336_rag"
    db_user: str = "cs336"
    db_password: str = "cs336"

    # Data layout
    data_dir: Path = Path("data")
    playlist_id: str = CS336_PLAYLIST_ID

    @property
    def db_dsn(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def raw_transcripts_dir(self) -> Path:
        return self.data_dir / "raw"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
