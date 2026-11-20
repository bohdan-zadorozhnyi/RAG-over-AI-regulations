"""One subcommand per stage. Stages never call each other."""

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def ingest() -> None:
    """Download source documents to data/raw/."""
    raise NotImplementedError


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
