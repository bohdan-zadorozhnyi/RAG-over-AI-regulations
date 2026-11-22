"""All tunables. Nothing else in the codebase holds a literal for these."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, model_validator
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
    # Names a question may use for a regulation, matched case-insensitively
    # on word boundaries to resolve "Article 12 of DORA" to a provision.
    document_aliases: dict[str, list[str]] = {
        "32024R1689": ["AI Act", "Artificial Intelligence Act", "Regulation (EU) 2024/1689"],
        "32016R0679": ["GDPR", "General Data Protection Regulation", "Regulation (EU) 2016/679"],
        "32022R2554": ["DORA", "Digital Operational Resilience Act", "Regulation (EU) 2022/2554"],
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
    # Chunks of explicitly referenced provisions considered (the third ranking).
    reference_top_k: int = 20
    # Reserve the results' tail for the best chunk of each regulation the
    # first fusion_top_k - (regulations - 1) results leave out.
    regulation_coverage: bool = True
    # Candidates per regulation offered to coverage. Without a reranker the
    # first one takes the slot; with one, the best by reranker score.
    coverage_candidates: int = 5

    # Reranking: a cross-encoder rescores the top rerank_candidates fused
    # chunks (plus the coverage candidates) before the top fusion_top_k are
    # kept. Runs on embedding_device, in fp32. Off by default: empty disables
    # it. Enable with REGRAG_RERANK_MODEL=BAAI/bge-reranker-v2-m3 (measured:
    # hit@5 0.867 -> 0.933, but search 31 ms -> 4.3 s on an M1 Pro, so it
    # needs a GPU host to be interactive).
    rerank_model: str = ""
    rerank_candidates: int = 40
    # Question + header + chunk; chunks are at most chunk_target_tokens.
    rerank_max_length: int = 1024
    rerank_batch_size: int = 16
    rrf_k: int = 60
    # ts_rank normalisation bitmask (Postgres docs): 0 none, 1 divide by
    # 1 + log(length), 2 divide by length. Long recitals win under 0; an eval
    # matrix dimension, not a judgement call.
    fts_rank_normalization: int = 0
    # Recitals explain intent but are not binding, and their explanatory prose
    # is closer to a plain-language question than article text, so they crowd
    # articles out. Excluding them won in all 6 strategy x norm pairs of the
    # step 6 baseline (14 questions gained, 3 lost). Still a matrix dimension.
    exclude_recitals: bool = True
    # HNSW candidates examined per vector search (pgvector default 40). Must be
    # at least vector_top_k; iterative scans extend it when filters drop rows.
    hnsw_ef_search: int = 100

    # Generation. Provider "anthropic" is the Claude API; Bedrock is the
    # intended production path once the AWS account has model quota (step 8).
    answer_provider: Literal["anthropic", "none"] = "anthropic"
    # Cheapest current model while the prompt is iterated; the answer eval
    # decides the shipped one.
    answer_model: str = "claude-haiku-5-5"
    # From .env (REGRAG_ANTHROPIC_API_KEY), never from the shell: Claude Code
    # would pick up a plain ANTHROPIC_API_KEY and bill its own usage to it.
    anthropic_api_key: SecretStr | None = None
    # Thinking depth (adaptive thinking is always on for current models).
    answer_effort: Literal["low", "medium", "high"] = "medium"
    # Room for thinking plus the reply; a reply cut off here is an error.
    answer_max_tokens: int = 8000
    # USD per million tokens (input, output), Claude API list prices, for
    # cost_usd. A model not listed is logged with no cost.
    answer_prices: dict[str, tuple[float, float]] = {
        "claude-haiku-5-5": (0.10, 0.50),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-opus-5-5": (4.00, 20.00),
    }
    # What the model reads: each retrieved provision expanded to its whole
    # article or annex (chars, about 4 per token). A longer one is cut to the
    # retrieved paragraphs and their neighbours; so is any provision once the
    # whole context would pass answer_context_max_chars.
    answer_article_max_chars: int = 8000
    answer_context_max_chars: int = 48000
    # Replies cached by everything that shapes them (model, effort, prompts),
    # so re-scoring an answer eval costs nothing. Under data/processed: not
    # committed.
    answer_cache_dir: Path = Path("data/processed/answer_cache")

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
        "rerank_batch_size",
        "embedding_device",
        "answer_provider",
        "answer_model",
        "anthropic_api_key",
        "answer_effort",
        "answer_max_tokens",
        "answer_prices",
        "answer_cache_dir",
        "answer_article_max_chars",
        "answer_context_max_chars",
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
