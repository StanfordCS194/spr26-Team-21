# Aperture — developer convenience targets.
# Run from the repo root: `make test`, `make serve`, `make build`, etc.

.PHONY: help install test test-cov serve build run docker-build docker-run clean

PYTHON ?= python3

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:             ## Install backend dependencies via Poetry
	cd backend && poetry install --no-interaction

test:                ## Run the pytest suite
	cd backend && poetry run pytest tests/ -v --tb=short

test-cov:            ## Run tests with coverage report
	cd backend && poetry run pytest tests/ --cov=services --cov-report=term-missing

serve:               ## Start the FastAPI dev server on :8000
	cd backend && poetry run uvicorn main:app --reload --port 8000

docker-build:        ## Build the backend Docker image
	cd backend && docker build -t aperture-backend .

docker-run:          ## Run the backend image and expose :8000
	docker run --rm -p 8000:8000 aperture-backend

clean:               ## Remove caches + .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
