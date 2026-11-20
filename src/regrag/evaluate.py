"""Step 6. The component that makes this engineering rather than a demo.

Metrics:
  hit_at_5  — any expected article in the top 5
  mrr       — mean reciprocal rank of the first expected article
  refusal_accuracy — out_of_scope questions correctly refused

Writes a row to eval_runs with git sha and config hash, per-question rows to
eval_results, and two files: a markdown report for humans and a flat JSON of
the metrics for machines. CI reads hit_at_5 from the JSON and fails the build
below the threshold; any future dashboard reads the same file.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Metrics:
    hit_at_5: float
    mrr: float
    refusal_accuracy: float
    n_questions: int
    by_category: dict[str, float]


def run(
    strategy: str,
    report_path: Path | None = None,
    metrics_path: Path | None = None,
) -> Metrics:
    """Score one strategy. Writes the report and metrics JSON when paths given."""
    raise NotImplementedError


def load_questions(path: Path) -> None:
    """Upsert data/eval/questions.yaml into eval_questions."""
    raise NotImplementedError


def write_metrics_json(metrics: dict[str, Metrics], gated: str, path: Path) -> None:
    """Flat JSON: top-level hit_at_5, mrr, refusal_accuracy, n_questions for the
    gated strategy (what CI checks), plus a `by_strategy` object for the rest."""
    raise NotImplementedError


def write_report(metrics: dict[str, Metrics], path: Path) -> None:
    """One table per strategy, so the comparison is the headline."""
    raise NotImplementedError
