"""Step 7. Answer with citations, or refuse.

A refusal is a correct output. If the retrieved provisions do not support an
answer, say so — most RAG demos get this wrong and it is the first thing a
reviewer tests.

Retrieve small, read big. Chunks are about a paragraph (they rank precisely),
but a paragraph rarely answers a question alone: "what does Article 35
require" retrieved paragraph 11 and none of 1-10. So each retrieved
provision is expanded to its whole article or annex before the model reads
it; one longer than answer_article_max_chars is cut to the retrieved
paragraphs and their neighbours, marked as an excerpt. Retrieval itself, and
what it is measured on, is unchanged.

The model sees these provisions as numbered sources and returns a JSON reply
(structured output, so it always parses): the answer with [n] markers, what
the sources leave out, and a status. Citations are then derived here, not
taken from the model: a marker resolves to a retrieved provision or is
dropped. An answer with no resolvable citation is turned into a
refusal, because CLAUDE.md's rule is that every answer cites an article.

Refusal is the model's judgement, not a similarity threshold: the eval showed
out-of-scope questions whose nearest chunk is closer than any in-scope one.

The generator is injectable (`Generator`): tests and CI use a fake, so
neither calls the API.
"""

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal, get_args

from psycopg.types.json import Jsonb

from regrag.config import settings
from regrag.db import Connection, connect
from regrag.encoder import Encoder
from regrag.models import Retrieved
from regrag.reranker import Reranker
from regrag.retrieve import SearchParams, search

Status = Literal["answer", "scoped", "refuse"]
STATUSES: tuple[Status, ...] = get_args(Status)

SYSTEM_PROMPT = """\
You answer questions about three EU regulations: the AI Act (Regulation (EU) \
2024/1689), the GDPR (Regulation (EU) 2016/679) and DORA (Regulation (EU) \
2022/2554). You answer only from the numbered sources given with each \
question: provisions retrieved from those three regulations, each a whole \
article or annex, or an excerpt of one where marked excerpt="true". They \
may not cover everything the question needs, and some may be irrelevant.

Base every statement on the sources. Do not add knowledge of your own about \
these or any other laws, even where you are confident it is right: a reader \
must be able to check each sentence against the source it cites. The three \
regulations are the whole scope. A question about another law (another EU \
act, national law, case law) is out of scope even when a source mentions or \
cross-refers to that law, because the sources do not contain its provisions.

Cite with source numbers in square brackets after the sentence or clause they \
support, for example [2] or [1][4]. Cite only sources that state what you \
attribute to them. Name the provision in the prose as well, as in "Article 32 \
GDPR requires...". Keep the conditions, exceptions and addressees the sources \
attach to a rule (who must do it, and when); they are usually the answer.

Set status:
- "answer": the sources answer the whole question.
- "scoped": the sources answer part of it. Answer that part, and in "missing" \
say plainly which part the sources do not cover, for example a regulation the \
question asks about whose relevant provision is not among the sources.
- "refuse": the sources answer no part of the question. Leave "answer" empty \
and say in "missing" what would be needed.

If the sources answer part of the question, for example the EU-level rule \
when the question asks how one Member State applies it, the status is \
"scoped", not "refuse": give that part in "answer", with citations. Anything \
you cite belongs in "answer", never in "missing".

Write plain, direct English, a short paragraph or a few bullet points. Do not \
hedge an answer the sources support, and do not add disclaimers about legal \
advice.\
"""

# Answer and gap come before the verdict, so the status follows from them.
REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "missing": {"type": "string"},
        "status": {"type": "string", "enum": list(STATUSES)},
    },
    "required": ["answer", "missing", "status"],
    "additionalProperties": False,
}

UNCITED = (
    "The model answered without citing a retrieved source, so the answer is "
    "withheld: every answer must cite the provision it rests on."
)
NOTHING_RETRIEVED = "No provisions were retrieved for this question."
MODEL_DECLINED = "The model declined to answer this request."

# Shorter shared starts are coincidence ("1. The"), not a repeated lead-in.
_MIN_LEAD_IN = 30

_MARKER = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")

# Every non-recital section of the given provisions, in document order.
PROVISION_SECTIONS = """
SELECT d.celex || ':' || s.article_no, s.id, s.heading, s.text
FROM sections s JOIN documents d ON d.id = s.document_id
WHERE s.kind <> 'recital' AND d.celex || ':' || s.article_no = ANY(%(provisions)s)
ORDER BY s.id
"""

# The sections each retrieved chunk overlaps.
RETRIEVED_SECTIONS = """
SELECT section_id FROM chunk_sections WHERE chunk_id = ANY(%(chunk_ids)s)
"""

INSERT_QUERY_LOG = """
INSERT INTO query_log
    (question, latency_ms, retrieved, answer, cost_usd,
     status, citations, model, input_tokens, output_tokens)
VALUES
    (%(question)s, %(latency_ms)s, %(retrieved)s, %(answer)s, %(cost_usd)s,
     %(status)s, %(citations)s, %(model)s, %(input_tokens)s, %(output_tokens)s)
RETURNING id
"""


class AnswerError(RuntimeError):
    """Generation failed: no key, an API error, or an unusable reply."""


@dataclass(frozen=True)
class Generation:
    """One model call: the raw JSON reply and what it cost."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


# (system prompt, user prompt) -> Generation
Generator = Callable[[str, str], Generation]


@dataclass(frozen=True)
class Source:
    """What the model reads for one retrieved provision."""

    provision: str  # "32016R0679:35"
    heading: str | None
    text: str
    excerpt: bool  # only some of the provision's paragraphs


# (section id, heading, text), in document order
SectionRow = tuple[int, str | None, str]


@dataclass(frozen=True)
class Answer:
    question: str
    status: Status
    text: str  # with [n] markers; n indexes `sources` from 1
    missing: str  # what the sources do not cover; "" when nothing
    citations: list[str]  # provisions, "32016R0679:32", in order of first citation
    cited_sources: list[int]  # resolvable source numbers, in order of first citation
    invalid_sources: list[int]  # markers naming no source; dropped
    downgraded: bool  # the model answered without a valid citation; refused here
    hits: list[Retrieved] = field(repr=False)  # what retrieval returned
    sources: list[Source] = field(repr=False)  # what the model read
    model: str | None  # None: no model call (nothing retrieved)
    input_tokens: int
    output_tokens: int
    latency_ms: int  # retrieval + generation
    cost_usd: float | None  # None: no price configured for the model

    @property
    def refused(self) -> bool:
        return self.status == "refuse"


# --- prompt -----------------------------------------------------------------


def provision_label(provision: str) -> str:
    """'32016R0679:32' -> 'GDPR Article 32'; annexes and recitals keep their name."""
    celex, _, number = provision.partition(":")
    name = settings.document_titles.get(celex, celex)
    if number.startswith(("Annex", "Recital")):
        return f"{name} {number}"
    return f"{name} Article {number}"


def provisions_in_order(hits: Sequence[Retrieved]) -> list[str]:
    """Each provision the hits cite, once, in the order retrieval ranked it."""
    seen: list[str] = []
    for hit in hits:
        for provision in hit.citations:
            if provision not in seen:
                seen.append(provision)
    return seen


def _shared_prefix(a: str, b: str) -> int:
    """Length of the common start of two strings."""
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b))
    )


def join_sections(texts: Sequence[str]) -> str:
    """Join consecutive sections, dropping a lead-in a section repeats from the
    one before it: the parser repeats "The Board shall ...: " on every point so
    each reads on its own, which whole-article text does not need."""
    parts: list[str] = []
    for i, text in enumerate(texts):
        if i:
            shared = _shared_prefix(texts[i - 1], text)
            if shared >= _MIN_LEAD_IN:
                cut = text.rfind(" ", 0, shared)  # don't split a word or label
                if cut > 0:
                    text = text[cut + 1 :]
        parts.append(text)
    return "\n".join(parts)


def excerpt(rows: Sequence[SectionRow], retrieved: set[int]) -> str:
    """The retrieved sections and one neighbour each side; gaps marked [...]."""
    keep = sorted(
        {j for i, row in enumerate(rows) if row[0] in retrieved for j in (i - 1, i, i + 1)}
        & set(range(len(rows)))
    )
    runs: list[list[str]] = []
    for prev, i in zip([None, *keep], keep, strict=False):
        if prev is None or i != prev + 1:
            runs.append([])
        runs[-1].append(rows[i][2])
    return "\n[...]\n".join(join_sections(run) for run in runs)


def build_sources(
    hits: Sequence[Retrieved],
    sections: Mapping[str, Sequence[SectionRow]],
    retrieved: set[int],
) -> list[Source]:
    """One source per retrieved provision, in retrieval order. Pure.

    `sections` holds each provision's sections; `retrieved` the section ids
    the hits overlap. A provision without sections (a recital) is read as its
    retrieved chunks. A whole provision is used while it fits both caps;
    otherwise the excerpt around what was retrieved.
    """
    out: list[Source] = []
    used = 0
    for provision in provisions_in_order(hits):
        rows = sections.get(provision, ())
        if not rows:
            text = "\n".join(h.text for h in hits if provision in h.citations)
            out.append(Source(provision, None, text, excerpt=True))
            used += len(text)
            continue
        heading = rows[0][1]
        whole = join_sections([row[2] for row in rows])
        fits = (
            len(whole) <= settings.answer_article_max_chars
            and used + len(whole) <= settings.answer_context_max_chars
        )
        source = (
            Source(provision, heading, whole, excerpt=False)
            if fits
            else Source(provision, heading, excerpt(rows, retrieved), excerpt=True)
        )
        out.append(source)
        used += len(source.text)
    return out


def sources_from_chunks(hits: Sequence[Retrieved]) -> list[Source]:
    """No expansion: each provision read as its retrieved chunks."""
    return build_sources(hits, {}, set())


def expand(conn: Connection, hits: Sequence[Retrieved]) -> list[Source]:
    """Load the sections of every retrieved provision, then build_sources."""
    sections: dict[str, list[SectionRow]] = {}
    for provision, section_id, heading, text in conn.execute(
        PROVISION_SECTIONS, {"provisions": provisions_in_order(hits)}
    ):
        sections.setdefault(provision, []).append((section_id, heading, text))
    retrieved = {
        row[0]
        for row in conn.execute(RETRIEVED_SECTIONS, {"chunk_ids": [h.chunk_id for h in hits]})
    }
    return build_sources(hits, sections, retrieved)


def build_prompt(question: str, sources: Sequence[Source]) -> str:
    """Sources first, numbered from 1, then the question (long context before
    the ask)."""
    blocks = []
    for n, source in enumerate(sources, start=1):
        label = provision_label(source.provision)
        title = f"{label}: {source.heading}" if source.heading else label
        flag = ' excerpt="true"' if source.excerpt else ""
        blocks.append(
            f'<source id="{n}" provision="{label}"{flag}>\n{title}\n{source.text}\n</source>'
        )
    return "<sources>\n" + "\n".join(blocks) + f"\n</sources>\n\n<question>{question}</question>"


# --- reply --------------------------------------------------------------------


def cited_source_numbers(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Markers in order of first appearance: (valid, invalid). Valid means
    1..n_sources; "[2, 3]" counts as two markers."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in _MARKER.finditer(text):
        for part in re.split(r"[,;]", match.group(1)):
            n = int(part)
            bucket = valid if 1 <= n <= n_sources else invalid
            if n not in bucket:
                bucket.append(n)
    return valid, invalid


def extract_citations(text: str, sources: Sequence[Source]) -> list[str]:
    """Only citations traceable to a retrieved provision count. Drop the rest."""
    valid, _ = cited_source_numbers(text, len(sources))
    return [sources[n - 1].provision for n in valid]


def parse_reply(generation: Generation) -> tuple[Status, str, str]:
    """(status, answer, missing). A safety decline is a refusal; a truncated or
    malformed reply is an error, never a guess."""
    if generation.stop_reason == "refusal":
        return "refuse", "", MODEL_DECLINED
    if generation.stop_reason == "max_tokens":
        raise AnswerError("reply truncated at answer_max_tokens; raise it")
    try:
        data = json.loads(generation.text)
    except json.JSONDecodeError as exc:
        raise AnswerError(f"reply is not JSON: {generation.text[:200]!r}") from exc
    if not isinstance(data, dict) or data.get("status") not in STATUSES:
        raise AnswerError(f"reply does not match the schema: {generation.text[:200]!r}")
    status: Status = data["status"]
    return status, str(data.get("answer", "")).strip(), str(data.get("missing", "")).strip()


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = settings.answer_prices.get(model)
    if prices is None:
        return None
    per_input, per_output = prices
    return (input_tokens * per_input + output_tokens * per_output) / 1_000_000


def compose(
    question: str,
    hits: Sequence[Retrieved],
    sources: Sequence[Source],
    generation: Generation | None,
    latency_ms: int,
) -> Answer:
    """Turn a reply into an Answer. Pure: everything the rules decide is here."""
    if generation is None:
        return Answer(
            question, "refuse", "", NOTHING_RETRIEVED, [], [], [], False,
            list(hits), list(sources), None, 0, 0, latency_ms, 0.0,
        )  # fmt: skip
    status, text, missing = parse_reply(generation)
    valid, invalid = cited_source_numbers(text, len(sources))
    citations = extract_citations(text, sources)
    downgraded = status != "refuse" and not citations
    if downgraded:
        status, missing = "refuse", UNCITED
    if status == "refuse":
        # A refusal states no law, so it cites none.
        valid, citations = [], []
    return Answer(
        question=question,
        status=status,
        text=text,
        missing=missing,
        citations=citations,
        cited_sources=valid,
        invalid_sources=invalid,
        downgraded=downgraded,
        hits=list(hits),
        sources=list(sources),
        model=generation.model,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd(generation.model, generation.input_tokens, generation.output_tokens),
    )


# --- generation ---------------------------------------------------------------


@cache
def load_generator() -> Generator:
    """The configured provider's model, created once per process."""
    if settings.answer_provider == "none":
        raise AnswerError("answer_provider is 'none'; use ask --retrieve-only")
    if settings.anthropic_api_key is None:
        raise AnswerError(
            "no API key: put REGRAG_ANTHROPIC_API_KEY=sk-ant-... in .env "
            "(not ANTHROPIC_API_KEY in the shell)"
        )
    import anthropic  # deferred: retrieval-only commands never need it

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    def generate(system: str, user: str) -> Generation:
        try:
            response = client.messages.create(
                model=settings.answer_model,
                max_tokens=settings.answer_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": settings.answer_effort,
                    "format": {"type": "json_schema", "schema": REPLY_SCHEMA},
                },
            )
        except anthropic.APIStatusError as exc:
            raise AnswerError(
                f"{type(exc).__name__} ({exc.status_code}): {exc.message} "
                f"[request {exc.request_id}]"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise AnswerError(f"could not reach the API: {exc}") from exc
        text = next((b.text for b in response.content if b.type == "text"), "")
        return Generation(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )

    return generate


def answer(
    question: str,
    params: SearchParams | None = None,
    conn: Connection | None = None,
    encoder: Encoder | None = None,
    reranker: Reranker | None = None,
    generate: Generator | None = None,
    log: bool = True,
) -> Answer:
    """Retrieve, generate, and (by default) record the exchange in query_log.

    Nothing retrieved means nothing to answer from: a refusal, with no call.
    """
    started = time.perf_counter()
    owns_conn = conn is None
    db = conn or connect()
    try:
        result = search(question, params, db, encoder, reranker)
        sources = expand(db, result.hits) if result.hits else []
        generation = None
        if sources:
            generation = (generate or load_generator())(
                SYSTEM_PROMPT, build_prompt(question, sources)
            )
        reply = compose(
            question,
            result.hits,
            sources,
            generation,
            round((time.perf_counter() - started) * 1000),
        )
        if log:
            log_query(db, reply)
    finally:
        if owns_conn:
            db.close()
    return reply


def log_query(conn: Connection, reply: Answer) -> int:
    row = conn.execute(
        INSERT_QUERY_LOG,
        {
            "question": reply.question,
            "latency_ms": reply.latency_ms,
            "retrieved": Jsonb(
                [
                    {"rank": rank, "chunk_id": h.chunk_id, "citations": list(h.citations)}
                    for rank, h in enumerate(reply.hits, start=1)
                ]
            ),
            "answer": reply.text or reply.missing,
            "cost_usd": reply.cost_usd,
            "status": reply.status,
            "citations": reply.citations,
            "model": reply.model,
            "input_tokens": reply.input_tokens,
            "output_tokens": reply.output_tokens,
        },
    ).fetchone()
    if row is None:
        raise RuntimeError("query_log insert returned no id")
    return int(row[0])
