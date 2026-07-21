"""Application settings, loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from cs336_rag.models import SearchMethod

CS336_PLAYLIST_ID = "PLoROMvodv4rMqXOcazWaTUHhq-yembLCV"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # OpenAI-compatible API (NaN)
    openai_key: str | None = None
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

    # Chunking / embedding batches
    chunk_max_chars: int = Field(default=1800, gt=0)
    chunk_overlap_chars: int = Field(default=300, ge=0)
    embed_batch_size: int = Field(default=64, gt=0)

    # Retrieval method served by default (winner of the retrieval evaluation;
    # see docs/evaluation.md). Override via RETRIEVAL_METHOD.
    retrieval_method: SearchMethod = "vector"

    # RAG answer generation. rag_prompt_variant is the winner of the answer
    # evaluation (see docs/evaluation.md); rag_context_size is how many chunks
    # are passed to the answer prompt.
    rag_prompt_variant: str = "grounded"
    rag_context_size: int = Field(default=5, gt=0)

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
    return Settings()
