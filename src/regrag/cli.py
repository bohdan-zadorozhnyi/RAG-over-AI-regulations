"""One subcommand per stage. Stages never call each other."""

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def ingest(
    refresh: bool = typer.Option(False, help="Re-fetch even if the file on disk is valid.")
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
        typer.echo(
            f"{r.status:<10} {d.celex} {d.title:<7} {d.checksum[:12]} {d.size_bytes:>10,} B"
        )


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
def embed(strategy: str = "section_aware") -> None:
    """Generate and store embeddings for pending chunks."""
    raise NotImplementedError


@app.command()
def index() -> None:
    """Build HNSW and GIN indexes."""
    from regrag.index import build_indexes

    build_indexes()


@app.command()
def ask(question: str) -> None:
    """Retrieve and answer with citations."""
    raise NotImplementedError


@app.command()
def eval(
    strategy: str = "all",
    report: str | None = None,
    metrics: str = "eval-metrics.json",
) -> None:
    """Run the eval set, write a markdown report and a flat metrics JSON."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
