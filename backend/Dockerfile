# Aperture backend image.
# Build:   docker build -t aperture-backend .
# Run:     docker run --rm -p 8000:8000 aperture-backend
# Health:  curl http://localhost:8000/health
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.0.0 \
    POETRY_VIRTUALENVS_CREATE=false

# Build deps for xgboost / scikit-learn wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Dependency layer (cached unless lock changes).
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi --without dev || \
    poetry install --no-interaction --no-ansi

# Application layer.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
