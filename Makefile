.PHONY: db migrate ingest parse chunk embed index ask eval eval-heldout check fmt clean

DB ?= postgresql://regrag:regrag@localhost:5432/regrag

db:
	docker compose up -d postgres
	@until docker compose exec -T postgres pg_isready -U regrag >/dev/null 2>&1; do sleep 1; done
	$(MAKE) migrate

# Override in CI: make migrate PSQL="psql $$REGRAG_DATABASE_URL"
PSQL ?= docker compose exec -T postgres psql -U regrag -d regrag

# Each file is applied and recorded in schema_migrations in one transaction;
# files already recorded are skipped. 000 is the idempotent bootstrap.
migrate:
	@$(PSQL) -v ON_ERROR_STOP=1 -q -f - < sql/000_schema_migrations.sql
	@for f in sql/[0-9]*.sql; do \
		name=$$(basename $$f); \
		applied=$$($(PSQL) -tAq -c "SELECT 1 FROM schema_migrations WHERE filename = '$$name'"); \
		if [ "$$applied" = "1" ]; then echo "skip   $$name"; continue; fi; \
		echo "apply  $$name"; \
		{ cat $$f; echo; echo "INSERT INTO schema_migrations (filename) VALUES ('$$name');"; } \
			| $(PSQL) -v ON_ERROR_STOP=1 -q --single-transaction -f - || exit 1; \
	done

ingest: ; uv run regrag ingest
parse:  ; uv run regrag parse
chunk:  ; uv run regrag chunk
embed:  ; uv run regrag embed
index:  ; uv run regrag index

ask:
	@test -n "$(Q)" || (echo 'usage: make ask Q="your question"'; exit 1)
	uv run regrag ask "$(Q)"

eval:   ; uv run regrag eval --report eval-report.md --metrics eval-metrics.json

# Held-out set: run ONCE, on the frozen shipped configuration only (no matrix,
# so there is nothing to pick a winner from). See data/eval/heldout.yaml.
eval-heldout:
	REGRAG_EVAL_QUESTIONS_PATH=data/eval/heldout.yaml REGRAG_EVAL_QUESTIONS_PER_CATEGORY=3 \
	uv run regrag eval --no-matrix --report eval-heldout-report.md --metrics eval-heldout-metrics.json

all: db ingest parse chunk embed index

check:
	uv run ruff check src tests
	uv run mypy src
	uv run pytest -q

fmt: ; uv run ruff format src tests && uv run ruff check --fix src tests

clean: ; docker compose down -v && rm -rf data/raw/* data/processed/*
