"""Unit tests for encoder.py. Never load the real model (2.2 GB download)."""

import pytest
from pydantic import ValidationError

from regrag.config import Settings
from regrag.encoder import (
    SPECIAL_TOKENS,
    check_dimension,
    check_fits,
    chunk_input,
    query_input,
    resolve_device,
)


def test_chunk_input_joins_header_and_text() -> None:
    assert chunk_input("AI Act, Recital 12", "(12) The notion…") == (
        "AI Act, Recital 12\n(12) The notion…"
    )


def test_chunk_input_without_header_is_the_text() -> None:
    assert chunk_input("", "baseline chunk text") == "baseline chunk text"


def test_query_input_is_the_question_alone() -> None:
    assert query_input("  What does\nArticle 9   require? ") == "What does Article 9 require?"


def test_explicit_device_is_kept() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_device_resolves_to_a_real_device() -> None:
    assert resolve_device("auto") in {"mps", "cuda", "cpu"}


def test_matching_dimension_passes() -> None:
    check_dimension(1024, 1024)


@pytest.mark.parametrize("model_dim", [768, None])
def test_dimension_mismatch_explains_the_migration(model_dim: int | None) -> None:
    with pytest.raises(ValueError, match="needs a new migration"):
        check_dimension(model_dim, 1024)


def test_chunk_that_fits_passes() -> None:
    check_fits(1024 - SPECIAL_TOKENS, 1024)


def test_chunk_that_would_be_truncated_fails() -> None:
    with pytest.raises(ValueError, match="embedding_max_seq_length"):
        check_fits(1024 - SPECIAL_TOKENS + 1, 1024)


def test_config_rejects_unknown_device_and_precision() -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_device="tpu")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        Settings(embedding_precision="int8")  # type: ignore[arg-type]
