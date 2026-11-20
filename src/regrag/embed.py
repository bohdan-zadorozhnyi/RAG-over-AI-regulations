"""Step 4a. Generate and store embeddings.

Model name and dimension are stored on every row, so re-embedding with a
different model is a new row and both can be evaluated side by side.
Idempotent on (chunk_id, model).
"""


def embed_pending(strategy: str) -> int:
    """Embed chunks lacking a vector for the configured model. Returns count."""
    raise NotImplementedError


def encode(texts: list[str]) -> list[list[float]]:
    """sentence-transformers locally, or Bedrock when provider is bedrock."""
    raise NotImplementedError
