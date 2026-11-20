# Build plan

## Definition of done

1. `make ask Q="..."` returns an answer with article citations
2. `make eval` prints hit rate@5 and MRR over 40 questions, writes a report
3. CI runs the eval on a PR and fails the build on regression
4. README has an architecture diagram, metrics table, limitations section
5. One deployment run on AWS, screenshotted, then torn down

## Corpus

| Document | CELEX | Notes |
|---|---|---|
| EU AI Act | 32024R1689 | 113 articles, 13 annexes |
| GDPR | 32016R0679 | 99 articles |
| DORA | 32022R2554 | ICT resilience |

Fetch the XHTML rendering from EUR-Lex, not the PDF. The HTML carries
article structure as markup; the PDF does not.

## Schema

documents    (id, celex, title, source_url, retrieved_at, checksum)
sections     (id, document_id, path, kind, article_no, paragraph_no, heading, text)
chunks       (id, document_id, section_id, ordinal, header, text, token_count, tsv)
embeddings   (chunk_id, model, dim, vector)
eval_questions (id, question, expected_articles, category)
eval_runs    (id, run_at, git_sha, config_hash, metrics)
eval_results (run_id, question_id, first_hit_rank, hit_at_5, retrieved)
query_log    (id, at, question, latency_ms, retrieved, answer, cost_usd)

`kind` is one of recital, article, annex. Recitals explain intent but are
not binding, so excluding them is a real retrieval feature.

## Steps

| # | Step | Budget | Done when |
|---|---|---|---|
| 0 | Repo, tooling, Postgres up, schema applied | 0:30 | `make db` works |
| 1 | Acquire: download 3 documents with manifest (url, celex, timestamp, checksum) | 1:30 | 3 files in data/raw |
| 2 | Parse: XHTML to sections, hierarchy and kind preserved | 3:00 | ~1,500 sections |
| 3 | Chunk: section-aware splitter + context header; LangChain recursive splitter as second strategy | 1:30 | ~2,500 chunks, 2 strategies |
| 4 | Embed and index: batch embed, HNSW on vector, GIN on tsv | 1:30 | populated index |
| 5 | Retrieve: vector top-20, FTS top-20, RRF to top-8, optional kind filter | 2:00 | `make ask` returns chunks |
| 6 | Eval: 40 questions, both strategies, hit@5 and MRR | 2:00 | a number for the CV |
| 7 | Answer: citations and refusal path, Langfuse tracing | 1:30 | cited answers |
| 8 | CI and Terraform: eval gate, S3 + Aurora + IAM + Bedrock | 1:30 | green PR |
| 9 | README and one AWS deploy run, then destroy | 1:30 | shippable |

Step 2 is where this overruns. If parsing is still fighting at the 4-hour
mark, fall back to a simpler article regex and move on. An imperfect parser
with a working eval beats a perfect parser and nothing downstream.

## Eval set

Forty questions, ten per category, in `data/eval/questions.yaml`:

- `direct_lookup` — names an article explicitly; tests lexical retrieval
- `obligation` — asks what someone must do; tests semantic retrieval
- `cross_document` — spans two regulations; tests fusion
- `out_of_scope` — the corpus does not answer it; tests refusal

Each carries expected article identifiers. That is the ground truth.

## Cut list, in order

1. FastAPI endpoint and any UI
2. Langfuse (log to query_log instead)
3. Answer generation entirely
4. AWS deploy (stay on local Docker)

Retrieval + eval + CI is the irreducible core.
