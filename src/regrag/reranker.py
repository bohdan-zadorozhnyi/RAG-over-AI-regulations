"""The second-stage model: a cross-encoder that rescores retrieval candidates.

The embedding model encodes the question and each chunk separately, so a
paraphrase ("buys in a high-risk AI system instead of building one") has to
land near the article's own wording ("deployers") in vector space. A
cross-encoder reads the question and one candidate together and scores how
well the candidate answers it, which handles paraphrase far better; it is
too slow to run over the corpus, so it only reorders the candidates that
fusion and regulation coverage already found.

Candidates are given as the embedded text (header + body, chunk_input), so
the reranker sees which regulation and article a chunk belongs to. Questions
go through query_input, as for embedding.

The reranker is injectable (`Reranker`): tests and CI use a deterministic
fake, so neither downloads the model. An empty rerank_model disables it.
"""

import os
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from regrag.config import settings
from regrag.encoder import resolve_device

# (question, candidate texts) -> one score per candidate; higher is better.
Reranker = Callable[[str, Sequence[str]], list[float]]


@cache
def load_reranker() -> Reranker:
    """Load the configured cross-encoder once per process.

    Downloads it (about 2.2 GB for bge-reranker-v2-m3) on first use into the
    Hugging Face cache; once cached it loads offline, as load_encoder does.
    Runs in fp32 on the embedding device: scores feed eval numbers, which
    should be reproducible.
    """
    from sentence_transformers import CrossEncoder  # slow import

    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")  # see load_encoder
    device = resolve_device(settings.embedding_device)
    kwargs: dict[str, Any] = {"device": device, "max_length": settings.rerank_max_length}
    try:
        model = CrossEncoder(settings.rerank_model, local_files_only=True, **kwargs)
    except OSError:  # not cached yet
        model = CrossEncoder(settings.rerank_model, **kwargs)

    def rerank(question: str, candidates: Sequence[str]) -> list[float]:
        if not candidates:
            return []
        scores: Any = model.predict(
            [(question, text) for text in candidates],
            batch_size=settings.rerank_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(s) for s in scores]

    return rerank
