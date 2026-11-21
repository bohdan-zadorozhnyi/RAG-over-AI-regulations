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
def parse() -> None:
    """Raw documents to sections rows."""
    raise NotImplementedError


@app.command()
def chunk(strategy: str = "section_aware") -> None:
    """Sections to chunks rows."""
    raise NotImplementedError


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
