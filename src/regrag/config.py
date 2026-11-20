"""All tunables. Nothing else in the codebase holds a literal for these."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REGRAG_", env_file=".env")

    database_url: str = "postgresql://regrag:regrag@localhost:5432/regrag"

    # Corpus
    celex_ids: list[str] = ["32024R1689", "32016R0679", "32022R2554"]
    eurlex_html_url: str = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"

    # Chunking
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 60
    chunk_strategy: str = "section_aware"  # or "recursive" (LangChain baseline)

    # Embeddings
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32

    # Retrieval
    vector_top_k: int = 20
    fts_top_k: int = 20
    fusion_top_k: int = 8
    rrf_k: int = 60

    # Generation
    answer_provider: str = "bedrock"  # or "gemini", or "none"
    answer_model: str = "anthropic.claude-sonnet-4-5-20250929-v1:0"

    langfuse_enabled: bool = False


settings = Settings()


def config_hash() -> str:
    """Stable hash of retrieval-affecting settings, recorded on every eval run."""
    raise NotImplementedError
