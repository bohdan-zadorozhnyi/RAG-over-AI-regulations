"""All tunables. Nothing else in the codebase holds a literal for these."""

import hashlib
import json
from collections.abc import Mapping
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
    # ts_rank normalisation bitmask (Postgres docs): 0 none, 1 divide by
    # 1 + log(length), 2 divide by length. Long recitals win under 0; an eval
    # matrix dimension, not a judgement call.
    fts_rank_normalization: int = 0
    # Recitals explain intent but are not binding. Also an eval matrix dimension.
    exclude_recitals: bool = False
    # HNSW candidates examined per vector search (pgvector default 40). Must be
    # at least vector_top_k; iterative scans extend it when filters drop rows.
    hnsw_ef_search: int = 100

    # Generation
    answer_provider: str = "bedrock"  # or "gemini", or "none"
    answer_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"

    langfuse_enabled: bool = False

    # Eval (defines the eval, not a retrieval configuration: keep out of config_hash)
    eval_questions_path: Path = Path("data/eval/questions.yaml")
    eval_questions_per_category: int = 10
    # ts_rank flags the matrix compares. 32 is left out: rank/(rank+1) is
    # monotone, so it cannot change the order.
    eval_fts_rank_normalizations: list[int] = [0, 1, 2]

    @model_validator(mode="after")
    def _every_celex_has_a_title(self) -> "Settings":
        missing = set(self.celex_ids) - self.document_titles.keys()
        if missing:
            raise ValueError(f"document_titles missing for: {sorted(missing)}")
        return self


settings = Settings()


# Settings that cannot change what retrieval returns. Everything else is
# hashed, so a retrieval setting added later changes the hash by default
# instead of being silently left out. Device is excluded as a property of the
# machine; fp32 vectors differ across devices only in the last bits.
_NOT_HASHED = frozenset(
    {
        "database_url",
        "raw_dir",
        "http_timeout_s",
        "http_retries",
        "http_user_agent",
        "embedding_batch_size",
        "embedding_device",
        "answer_provider",
        "answer_model",
        "langfuse_enabled",
        "eval_questions_path",
        "eval_questions_per_category",
        "eval_fts_rank_normalizations",
    }
)


def config_hash(overrides: Mapping[str, object] | None = None) -> str:
    """Stable hash of retrieval-affecting settings, recorded on every eval run.

    `overrides` replaces setting values first: an eval matrix combination is
    the shipped settings with a few fields changed.
    """
    values = settings.model_dump(mode="json", exclude=set(_NOT_HASHED))
    unknown = set(overrides or {}) - values.keys()
    if unknown:
        raise ValueError(f"not hashed settings: {sorted(unknown)}")
    values.update(overrides or {})
    encoded = json.dumps(values, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]
