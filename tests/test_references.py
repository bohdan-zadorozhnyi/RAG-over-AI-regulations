"""parse_references: explicit provision references to citations. No database."""

import pytest

from regrag.config import settings
from regrag.retrieve import parse_references

AI_ACT, GDPR, DORA = "32024R1689", "32016R0679", "32022R2554"


@pytest.fixture(autouse=True)
def corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "celex_ids", [AI_ACT, GDPR, DORA])


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does Article 12 of DORA require?", [f"{DORA}:12"]),
        ("Under DORA's Article 30, what goes in the contract?", [f"{DORA}:30"]),
        ("GDPR Art. 35 — what does it oblige?", [f"{GDPR}:35"]),
        ("GDPR art 35", [f"{GDPR}:35"]),
        ("What does Annex III of the AI Act list?", [f"{AI_ACT}:Annex III"]),
        ("Under the AI Act, what does Article 17 cover?", [f"{AI_ACT}:17"]),
        ("Is Article 6(2) of the Artificial Intelligence Act relevant?", [f"{AI_ACT}:6"]),
        (
            "Compare Article 22 GDPR with Article 86 of the AI Act",
            [f"{GDPR}:22", f"{AI_ACT}:86"],
        ),
        ("What rights does the General Data Protection Regulation give?", []),
        ("Which practices are prohibited?", []),
    ],
)
def test_parse_references(question: str, expected: list[str]) -> None:
    assert parse_references(question) == expected


def test_unpaired_reference_applies_to_every_regulation_in_scope() -> None:
    assert parse_references("What does Article 9 say?") == [f"{AI_ACT}:9", f"{GDPR}:9", f"{DORA}:9"]


def test_words_containing_art_are_not_references() -> None:
    assert parse_references("Part 2 of the startup's 3 plans, apart 4") == []


def test_regulations_outside_scope_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "celex_ids", [GDPR])
    assert parse_references("Article 12 of DORA") == [f"{GDPR}:12"]  # DORA not searched
