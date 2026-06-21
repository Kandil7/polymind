.PHONY: dev test test-integration test-e2e eval lint format docs clean seed docker-build deploy

dev:
	docker compose up -d qdrant prometheus grafana
	poetry run uvicorn polymind.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest tests/unit tests/integration -v --tb=short

test-integration:
	poetry run pytest tests/integration -v --tb=short

test-e2e:
	poetry run pytest tests/e2e -v --tb=short

eval:
	poetry run pytest tests/eval -v --tb=short

lint:
	poetry run ruff check src/ tests/
	poetry run mypy src/polymind/

format:
	poetry run ruff format src/ tests/
	poetry run ruff check --fix src/ tests/

docs:
	@echo "Documentation files are in docs/"
	@echo "  - README.md: Project overview"
	@echo "  - docs/API_REFERENCE.md: API reference"
	@echo "  - docs/ROADMAP.md: Build phases"
	@echo "  - docs/architecture/: Architecture docs and ADRs"
	@echo "  - docs/learning/: Learning guides"
	@echo "  - docs/phases/: Phase documentation"

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true

seed:
	poetry run python scripts/seed_qdrant.py

docker-build:
	docker build -t polymind:latest -f infra/Dockerfile .

deploy:
	poetry run python infra/modal_deploy.py
