"""All tunables. Nothing else in the codebase holds a literal for these."""

from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGRAG_", env_file=".env")

    database_url: str = "postgresql://regrag:regrag@localhost:5432/regrag"

    # Corpus
    # celex_ids sets order and allows a subset; document_titles is the lookup.
    # Env overrides for lists and dicts are JSON: REGRAG_CELEX_IDS='["32024R1689"]'
    celex_ids: list[str] = ["32024R1689", "32016R0679", "32022R2554"]
    document_titles: dict[str, str] = {
        "32024R1689": "AI Act",
        "32016R0679": "GDPR",
        "32022R2554": "DORA",
    }
    eurlex_html_url: str = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

    # Acquisition (not retrieval-affecting: keep out of config_hash)
    raw_dir: Path = Path("data/raw")
    http_timeout_s: float = 30.0
    http_retries: int = 3  # connection-level retries only; HTTP error statuses fail loudly
    http_user_agent: str = "regrag/0.1"

    # Chunking
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 60
    chunk_strategy: str = "section_aware"  # or "recursive" (LangChain baseline)

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    # "auto" picks mps (Apple GPU), then cuda, then cpu.
    embedding_device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    # fp32 by default: fp16 is faster but its vectors vary slightly across runs
    # and machines, and eval numbers should be reproducible.
    embedding_precision: Literal["fp32", "fp16"] = "fp32"
    # Cap on tokens per input. bge-m3 accepts 8192; chunks are at most
    # chunk_target_tokens, so this is a guard against silent truncation.
    embedding_max_seq_length: int = 1024

    # Vector index (HNSW build parameters, see pgvector docs)
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64

    # Retrieval
    vector_top_k: int = 20
    fts_top_k: int = 20
    fusion_top_k: int = 8
    rrf_k: int = 60

    # Generation
    answer_provider: str = "bedrock"  # or "gemini", or "none"
    answer_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"

    langfuse_enabled: bool = False

    @model_validator(mode="after")
    def _every_celex_has_a_title(self) -> "Settings":
        missing = set(self.celex_ids) - self.document_titles.keys()
        if missing:
            raise ValueError(f"document_titles missing for: {sorted(missing)}")
        return self


settings = Settings()


def config_hash() -> str:
    """Stable hash of retrieval-affecting settings, recorded on every eval run."""
    raise NotImplementedError
