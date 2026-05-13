# Aperture — Backend

FastAPI server that powers schema inference, synthetic data generation, and fidelity validation.

## Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Data handling | pandas, pyarrow |
| Synthesis | SDV `GaussianCopulaSynthesizer` (file-grounded) · NumPy statistical sampler (NL-only) |
| NL → schema | Anthropic Claude `claude-sonnet-4-6` with keyword-matching fallback |
| Export formats | CSV, JSONL, Parquet |

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/) (Python 3.11–3.13).

```bash
cd backend
poetry install
```

Create a `.env` file (copy from `.env.example` if present):

```
ANTHROPIC_API_KEY=sk-ant-...
```

The API key is optional. Without it the `/api/plan` endpoint uses a deterministic keyword-matching fallback instead of Claude.

Start the dev server:

```bash
poetry run uvicorn main:app --reload --port 8000
```

## Project layout

```
backend/
├── main.py            # FastAPI app + CORS + router wiring
├── api/               # HTTP endpoints (one module per resource)
│   ├── health.py      # GET  /health
│   ├── plan.py        # POST /api/plan
│   ├── schema.py      # POST /api/infer-schema
│   └── generate.py    # POST /api/preview, /api/generate; GET /api/download/{id}
├── core/
│   ├── config.py      # .env, Anthropic client, SDV availability flags
│   ├── state.py       # in-memory session + SDV model stores
│   └── io.py          # upload file readers (csv / xlsx / parquet)
├── models/
│   └── schemas.py     # Pydantic request bodies
└── services/
    ├── inference.py   # column type / distribution / stats inference
    ├── synthesis.py   # statistical sampler + SDV-aware row generator
    ├── sdv_model.py   # GaussianCopula fit
    ├── validation.py  # realism / diversity / PII scoring
    └── planner.py     # NL → schema (system prompt + keyword fallback)
```

## Endpoints

### `GET /health`
Returns `{"status": "ok"}`. Use for readiness checks.

---

### `POST /api/plan`
Infer a dataset schema and generation plan from a natural language prompt.

**Request**
```json
{ "prompt": "Generate 10,000 diabetic patient records with HbA1c > 12 edge cases" }
```

**Response**
```json
{
  "schema": [{ "column": "age", "type": "int", "distribution": "Gaussian (μ=52, σ=8)", "sample": "54" }],
  "generation_spec": { "row_count": 10000, "format": "csv", "edge_cases": ["hba1c > 12"], "labels": [], "constraints": [] },
  "generation_plan": { "intro": "...", "steps": [{ "title": "Schema Design", "description": "..." }] },
  "clarifying_questions": []
}
```

When the prompt is underspecified, `clarifying_questions` contains 1–3 follow-up questions and `schema` contains a best-guess schema. When `ANTHROPIC_API_KEY` is not set, the fallback matches keywords (`patient`, `clinical`, `hba1c` → medical schema; `order`, `transaction` → e-commerce; etc.).

---

### `POST /api/infer-schema`
Upload real data files to profile and ground generation.

**Request** — multipart form, field name `files`. Accepts `.csv`, `.xlsx`, `.xls`, `.parquet`.

**Response**
```json
{
  "columns": [{ "column": "age", "type": "int", "distribution": "Gaussian (μ=52, σ=8)", "sample": "54", "null_pct": 0.0 }],
  "stats": { "age": { "col_type": "int", "mean": 52.1, "std": 8.3, "min": 18, "max": 89, "skew": 0.2 } },
  "source_rows": 50000,
  "model_id": "c3f1a2b4-..."
}
```

`null_pct` is the percentage of missing values for each column (0–100).

`model_id` is the ID of a fitted SDV `GaussianCopulaSynthesizer` stored server-side. Pass it to `/api/generate` and `/api/preview` to enable correlation-preserving synthesis. Returns `null` if fitting failed or SDV is unavailable.

**Inferred column types**

| Type | Detected when |
|---|---|
| `int` | pandas integer dtype |
| `float` | pandas float dtype |
| `bool` | pandas bool dtype |
| `date` | datetime dtype or parseable date strings |
| `uuid` | >80% of non-null values match UUID format |
| `enum` | string column with categorical values |
| `array<str>` | >50% of values start with `[` and end with `]` |
| `text` | string column that doesn't fit the above |

---

### `POST /api/preview`
Generate 10 synthetic rows without creating a download session. Useful for checking schema correctness before committing to full generation.

**Request** — same shape as `/api/generate`.

**Response**
```json
{ "rows": [{ "age": 54, "sex": "F", "hba1c": 7.2 }] }
```

---

### `POST /api/generate`
Generate a full synthetic dataset and run fidelity validation.

**Request**
```json
{
  "schema_columns": [{ "column": "age", "type": "int" }],
  "source_stats": { "age": { "col_type": "int", "mean": 52.1, "std": 8.3, "min": 18, "max": 89, "skew": 0.2 } },
  "row_count": 10000,
  "format": "csv",
  "prompt": "diabetic cohort",
  "edge_cases": [],
  "model_id": "c3f1a2b4-..."
}
```

`row_count` is clamped to 1–100,000. `format` must be `csv`, `jsonl`, or `parquet`. `model_id` is optional — omit it or pass `null` to use the statistical sampler.

**Response**
```json
{
  "session_id": "a1b2c3d4-...",
  "row_count": 10000,
  "file_size_kb": 842.3,
  "format": "csv",
  "filename": "aperture_output.csv",
  "validation": {
    "verdict": "Ready for use",
    "verdictStatus": "pass",
    "metrics": [
      { "label": "Realism",     "score": 94, "status": "pass" },
      { "label": "Diversity",   "score": 87, "status": "pass" },
      { "label": "Safety / PII","score": 100,"status": "pass" }
    ],
    "columns": [{ "column": "age", "fidelity": 96, "status": "pass" }],
    "insights": ["No PII detected across all 10,000 rows"]
  }
}
```

---

### `GET /api/download/{session_id}`
Stream the generated file for a previously completed generation. Sessions are stored in memory and lost on server restart.

---

## Synthesis

### SDV path (file-grounded)
When real data is uploaded, `_fit_sdv_model` fits a `GaussianCopulaSynthesizer` on up to 50,000 rows. The model captures the joint distribution of all synthesisable columns (numerical, categorical, datetime) so inter-column correlations are preserved in the output. `uuid` and `array<str>` columns are excluded from SDV and generated independently.

### Statistical sampler (NL-only fallback)
Used when no grounding data is available or SDV fitting fails. Each column is synthesised independently:
- `int`/`float` — `np.random.normal` or `np.random.lognormal` (chosen by skew > 1.5)
- `enum` — `np.random.choice` weighted by observed category frequencies
- `bool` — Bernoulli with inferred `p_true`
- `date` — uniform random timestamp between inferred min and max
- `uuid` — `uuid.uuid4()`

### Validation
After synthesis, `_validate` compares the output against `source_stats` from the uploaded file:
- **Realism** — mean fidelity across numeric columns; fidelity = `100 - (mean_drift + std_drift) * 50`
- **Diversity** — compares coefficient of variation between source and synthetic
- **Safety / PII** — regex scan for email, phone, and SSN patterns across all string columns

Validation scores are only meaningful when `source_stats` were derived from real uploaded data. When generating from a natural language prompt with no grounding file, the validation report shown in the UI is a placeholder.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | No | Enables Claude-powered `/api/plan`. Falls back to keyword matching if unset. |

## Known limitations

- Sessions (`_sessions`) and fitted SDV models (`_sdv_models`) are in-process Python dicts — both are lost on server restart.
- Edge cases specified in prompts (e.g. "HbA1c > 12 with 3+ comorbidities") are stored in `GenerationSpec.edge_cases` but not yet enforced during synthesis.
- `text` column synthesis produces placeholder strings (`val_0`, `val_1`, …) — real sentence generation is not implemented.
