.PHONY: install lint format format-check typecheck test check ci pre-commit pre-commit-install

UV ?= uv
UVX ?= uvx

CI_ENV = DB_HOST=localhost DB_PORT=5432 DB_NAME=cs336_rag DB_USER=cs336 DB_PASSWORD=cs336 OPENAI_KEY=test-key-not-used

install:
	$(UV) sync --frozen

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format --check .

typecheck:
	$(UV) run mypy

test:
	$(CI_ENV) $(UV) run pytest -v --cov --cov-report=term-missing

check: lint format-check typecheck test

ci: install check

pre-commit:
	$(UVX) pre-commit run --all-files

pre-commit-install:
	$(UVX) pre-commit install
