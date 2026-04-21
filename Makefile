.PHONY: install dev test lint format lint-fix clean run-api run-dashboard run-worker docker-up docker-down help

VENV = .venv
UV ?= uv

install: install-base install-dev          ## Install all dependencies
install-base:
	$(UV) sync
install-dev:
	$(UV) sync --all-extras

run-api:                     ## Run FastAPI server
	$(UV) run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard:              ## Run Streamlit dashboard
	$(UV) run streamlit run dashboard/app.py --server.port 8501 --server.headless true

run-worker:                 ## Run trading worker
	$(UV) run src/main.py worker

run-cli:                    ## Run CLI
	$(UV) run trade --help

test:                       ## Run tests
	$(UV) run pytest tests/ -v --tb=short

test-coverage:              ## Run tests with coverage
	$(UV) run pytest tests/ -v --cov=src --cov=api --cov=dashboard --cov=cli --cov-report=html --cov-report=term-missing

lint:                       ## Run linting
	$(UV) run ruff check src/ api/ dashboard/ cli/ tests/
	$(UV) run mypy src/ api/ dashboard/ cli/ tests/

lint-fix:                   ## Auto-fix linting issues
	$(UV) run ruff check --fix src/ api/ dashboard/ cli/ tests/

format:                     ## Format code
	$(UV) run ruff format src/ api/ dashboard/ cli/ tests/

clean:                      ## Clean build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	$(RM) -r dist/ build/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage

docker-up:                  ## Start all services with docker-compose
	docker-compose up -d --build
docker-down:                ## Stop all services
	docker-compose down -v

help:                       ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
