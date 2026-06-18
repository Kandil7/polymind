.PHONY: dev test test-integration eval lint format docs clean seed

dev:
	docker compose up -d qdrant prometheus grafana
	poetry run uvicorn polymind.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests/unit tests/integration -v --tb=short

test-integration:
	poetry run pytest tests/integration -v --tb=short

eval:
	poetry run pytest tests/eval -v --tb=short

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/polymind/

format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

docs:
	poetry run mkdocs serve

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true

seed:
	poetry run python scripts/seed_qdrant.py
