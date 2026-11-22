"""One subcommand per stage. Stages never call each other."""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def ingest(
    refresh: bool = typer.Option(False, help="Re-fetch even if the file on disk is valid."),
) -> None:
    """Download source documents to data/raw/."""
    import httpx

    from regrag.acquire import fetch_all
    from regrag.config import settings

    try:
        results = fetch_all(settings.raw_dir, refresh=refresh)
    except (httpx.HTTPError, ValueError) as exc:
        typer.echo(f"ingest failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for r in results:
        d = r.document
        typer.echo(f"{r.status:<10} {d.celex} {d.title:<7} {d.checksum[:12]} {d.size_bytes:>10,} B")


@app.command()
def parse(
    force: bool = typer.Option(False, help="Re-parse even if the stored checksum matches."),
) -> None:
    """Raw documents to sections rows."""
    import psycopg

    from regrag.config import settings
    from regrag.parse import parse_all

    try:
        results = parse_all(settings.raw_dir, force=force)
    except (FileNotFoundError, ValueError, psycopg.OperationalError) as exc:
        typer.echo(f"parse failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for r in results:
        kinds = ", ".join(f"{n} {k}" for k, n in sorted(r.by_kind.items()))
        detail = f"({kinds}) written {r.written}, deleted {r.deleted}" if kinds else ""
        typer.echo(f"{r.status:<8} {r.celex}  {r.sections:>5} sections  {detail}".rstrip())


@app.command()
def chunk(
    strategy: str = typer.Option("all", help="section_aware, recursive, or all."),
) -> None:
    """Sections to chunks rows."""
    import psycopg

    from regrag.chunk import chunk_all

    try:
        results = chunk_all(strategy)
    except (ValueError, psycopg.OperationalError) as exc:
        typer.echo(f"chunk failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for r in results:
        typer.echo(
            f"{r.strategy:<14} {r.celex}  {r.chunks:>5} chunks  max {r.max_tokens:>3} tokens  "
            f"inserted {r.inserted}, updated {r.updated}, deleted {r.deleted}"
        )


@app.command()
def embed(
    strategy: str = typer.Option("all", help="section_aware, recursive, or all."),
) -> None:
    """Generate and store embeddings for pending chunks."""
    import time

    import psycopg

    from regrag.config import settings
    from regrag.embed import embed_pending
    from regrag.encoder import resolve_device

    device = resolve_device(settings.embedding_device)
    typer.echo(f"model {settings.embedding_model} on {device}, {settings.embedding_precision}")

    def progress(done: int, total: int) -> None:
        typer.echo(f"\r  embedded {done}/{total}", nl=False)
        if done == total:
            typer.echo("")

    started = time.monotonic()
    try:
        results = embed_pending(strategy, progress=progress)
    except (ValueError, psycopg.OperationalError) as exc:
        typer.echo(f"\nembed failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    for r in results:
        typer.echo(f"{r.strategy:<14} embedded {r.embedded}, already had {r.already}")
    typer.echo(f"done in {time.monotonic() - started:.0f}s")


@app.command()
def index() -> None:
    """Build HNSW and GIN indexes."""
    from regrag.index import build_indexes

    build_indexes()


@app.command()
def ask(
    question: str,
    strategy: str | None = typer.Option(
        None, help="section_aware or recursive; defaults to the chunk_strategy setting."
    ),
    exclude_recitals: bool | None = typer.Option(
        None,
        "--exclude-recitals/--include-recitals",
        help="Search binding text only (articles, annexes), or everything. "
        "Defaults to the exclude_recitals setting.",
        show_default=False,
    ),
) -> None:
    """Retrieve and answer with citations.

    Until step 7 adds answer generation, prints the retrieved chunks.
    """
    import psycopg

    from regrag import retrieve

    try:
        params = retrieve.SearchParams.from_settings(
            strategy=strategy, exclude_recitals=exclude_recitals
        )
        result = retrieve.search(question, params)
    except (ValueError, psycopg.OperationalError) as exc:
        typer.echo(f"ask failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not result.hits:
        typer.echo("no chunks retrieved")
        return
    typer.echo(
        f"{len(result.hits)} chunks ({params.label}), "
        f"nearest similarity {_similarity(result.nearest_similarity)}:"
    )
    for rank, hit in enumerate(result.hits, start=1):
        typer.echo("")
        typer.echo(
            f"{rank}. {', '.join(hit.citations)}   score {hit.fused_score:.4f}   "
            f"vector #{_rank(hit.vector_rank)}  text #{_rank(hit.fts_rank)}   "
            f"sim {_similarity(hit.similarity)}"
        )
        if hit.header:
            typer.echo(f"   {hit.header}")
        typer.echo(f"   {_snippet(hit.text)}")


def _similarity(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _rank(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _snippet(text: str, limit: int = 220) -> str:
    """One line: list points joined with ' / ', cut at a word boundary."""
    flat = " / ".join(line.strip() for line in text.splitlines() if line.strip())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + " …"


@app.command()
def eval(
    strategy: str = typer.Option("all", help="section_aware, recursive, or all."),
    matrix: bool = typer.Option(
        True, "--matrix/--no-matrix", help="Every combination, or only the shipped settings."
    ),
    report: str | None = typer.Option(None, help="Markdown report path."),
    metrics: str = typer.Option("eval-metrics.json", help="Flat metrics JSON path."),
    allow_incomplete: bool = typer.Option(
        False, "--allow-incomplete", help="Run with fewer questions than the full set."
    ),
) -> None:
    """Run the eval set, write a markdown report and a flat metrics JSON."""
    import psycopg

    from regrag.evaluate import run

    try:
        runs = run(
            strategy,
            matrix=matrix,
            report_path=Path(report) if report else None,
            metrics_path=Path(metrics),
            allow_incomplete=allow_incomplete,
        )
    except (ValueError, FileNotFoundError, psycopg.OperationalError) as exc:
        typer.echo(f"eval failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    n = runs[0].metrics.n_questions
    typer.echo(f"{n} questions, {len(runs)} combinations")
    typer.echo(f"{'configuration':<38} {'hit@5':>6} {'recall@5':>9} {'MRR':>6} {'AUC':>6}")
    for r in runs:
        m = r.metrics
        auc = "-" if m.separation_auc is None else f"{m.separation_auc:.3f}"
        typer.echo(
            f"{r.params.label:<38} {m.hit_at_5:>6.3f} {m.recall_at_5:>9.3f} {m.mrr:>6.3f} {auc:>6}"
        )
    written = [p for p in (report, metrics) if p]
    typer.echo(f"wrote {', '.join(written)}")


if __name__ == "__main__":
    app()
