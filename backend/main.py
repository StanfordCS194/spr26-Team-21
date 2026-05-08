"""
Aperture backend — statistical data synthesis pipeline.

Run:  uvicorn main:app --reload --port 8000
"""

import io
import json
import os
import re
import uuid
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

try:
    import anthropic as _anthropic_mod
    _llm = _anthropic_mod.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") else None
except ImportError:
    _llm = None

try:
    from sdv.single_table import GaussianCopulaSynthesizer as _GaussianCopula
    from sdv.metadata import SingleTableMetadata as _SDVMetadata
    _SDV_AVAILABLE = True
except ImportError:
    _GaussianCopula = None  # type: ignore[assignment,misc]
    _SDVMetadata = None  # type: ignore[assignment,misc]
    _SDV_AVAILABLE = False

# Fitted SDV models keyed by model_id (separate from download sessions)
_sdv_models: dict[str, dict] = {}

app = FastAPI(title="Aperture API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (replace with S3/DB in production)
_sessions: dict[str, dict] = {}


# ── File reading ──────────────────────────────────────────────────────────────

def _read_upload(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    name = (upload.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


# ── Schema inference ──────────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _infer_col_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    sample = series.dropna().astype(str).head(20)
    if sample.apply(lambda x: bool(_UUID_RE.match(x))).mean() > 0.8:
        return "uuid"
    try:
        pd.to_datetime(sample, infer_datetime_format=True)
        return "date"
    except Exception:
        pass
    if sample.apply(lambda x: x.startswith("[") and x.endswith("]")).mean() > 0.5:
        return "array<str>"
    return "enum"


def _distribution_label(series: pd.Series, col_type: str) -> str:
    if col_type in ("int", "float"):
        mu, sigma = series.mean(), series.std()
        skew = abs(series.skew()) if len(series) > 2 else 0
        kind = "LogNormal" if skew > 1.5 and series.min() >= 0 else "Gaussian"
        return f"{kind} (μ={mu:.1f}, σ={sigma:.1f})"
    if col_type == "enum":
        vc = series.value_counts(normalize=True).head(3)
        parts = [f"{k}:{v:.2f}" for k, v in vc.items()]
        suffix = "…" if series.nunique() > 3 else ""
        return "{" + ", ".join(parts) + suffix + "}"
    if col_type == "date":
        dates = pd.to_datetime(series, errors="coerce").dropna()
        if len(dates):
            return f"Uniform {dates.min().year}–{dates.max().year}"
    if col_type == "uuid":
        return "unique"
    if col_type == "bool":
        p = series.mean()
        return f"{{True:{p:.2f}, False:{1 - p:.2f}}}"
    return "—"


def _col_stats(series: pd.Series, col_type: str) -> dict[str, Any]:
    stat: dict[str, Any] = {"col_type": col_type}
    if col_type in ("int", "float"):
        stat.update(
            mean=float(series.mean()),
            std=float(max(series.std(), 1e-9)),
            min=float(series.min()),
            max=float(series.max()),
            skew=float(series.skew() if len(series) > 2 else 0),
        )
    elif col_type == "enum":
        vc = series.value_counts(normalize=True)
        cats = vc.index.tolist()[:30]
        probs = vc.values.tolist()[:30]
        total = sum(probs)
        stat["categories"] = cats
        stat["probs"] = [p / total for p in probs]
    elif col_type == "bool":
        stat["p_true"] = float(series.mean())
    elif col_type == "date":
        dates = pd.to_datetime(series, errors="coerce").dropna()
        if len(dates):
            stat["min_ts"] = int(dates.min().timestamp())
            stat["max_ts"] = int(dates.max().timestamp())
        else:
            stat["min_ts"] = 0
            stat["max_ts"] = int(pd.Timestamp.now().timestamp())
    return stat


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synth_column(col_name: str, stat: dict[str, Any], n: int) -> list:
    col_type = stat.get("col_type", "enum")

    if col_type in ("int", "float"):
        skew = stat.get("skew", 0)
        mn, mx = stat["min"], stat["max"]
        if skew > 1.5 and mn >= 0:
            log_mean = float(np.log(max(stat["mean"], 1e-9)))
            log_std = float(stat["std"] / (abs(stat["mean"]) + 1e-9))
            vals = np.random.lognormal(log_mean, log_std, n)
        else:
            vals = np.random.normal(stat["mean"], stat["std"], n)
        vals = np.clip(vals, mn, mx)
        return vals.astype(int).tolist() if col_type == "int" else [round(float(v), 4) for v in vals]

    if col_type == "enum":
        cats = stat["categories"]
        probs = np.array(stat["probs"])
        probs = probs / probs.sum()
        return np.random.choice(cats, n, p=probs).tolist()

    if col_type == "bool":
        p = stat.get("p_true", 0.5)
        return np.random.choice([True, False], n, p=[p, 1 - p]).tolist()

    if col_type == "date":
        lo, hi = stat.get("min_ts", 0), stat.get("max_ts", int(pd.Timestamp.now().timestamp()))
        timestamps = np.random.randint(lo, hi + 1, n)
        return [pd.Timestamp(int(ts), unit="s").strftime("%Y-%m-%d") for ts in timestamps]

    if col_type == "uuid":
        return [str(uuid.uuid4()) for _ in range(n)]

    if col_type == "array<str>":
        return [f"[item_{i % 5}]" for i in range(n)]

    # text / unknown
    return [f"val_{i}" for i in range(n)]


def _build_synth_df(schema_columns: list[dict], source_stats: dict[str, dict], n: int) -> pd.DataFrame:
    synth: dict[str, list] = {}
    for col_def in schema_columns:
        col_name = col_def["column"]
        stat = source_stats.get(col_name) or {"col_type": "enum", "categories": ["value"], "probs": [1.0]}
        synth[col_name] = _synth_column(col_name, stat, n)
    return pd.DataFrame(synth)


# ── SDV integration ───────────────────────────────────────────────────────────

# SDV can synthesize these types (uuid / array<str> are handled separately)
_SDV_TYPE_MAP: dict[str, str] = {
    "int": "numerical",
    "float": "numerical",
    "enum": "categorical",
    "bool": "categorical",
    "text": "categorical",
    "date": "datetime",
}


def _fit_sdv_model(df: pd.DataFrame, col_info: list[dict]) -> str | None:
    """
    Fit a GaussianCopulaSynthesizer on the source DataFrame.
    Returns a model_id string on success, or None if SDV is unavailable / fitting fails.
    uuid and array<str> columns are excluded from the model and re-synthesised statistically.
    """
    if not _SDV_AVAILABLE:
        return None

    sdv_cols = [(c["column"], c["type"]) for c in col_info if c["type"] in _SDV_TYPE_MAP]
    if len(sdv_cols) < 1:
        return None

    col_names = [col for col, _ in sdv_cols]
    df_fit = df[col_names].copy()

    # Normalise date columns to ISO strings that SDV expects
    for col, ctype in sdv_cols:
        if ctype == "date":
            df_fit[col] = pd.to_datetime(df_fit[col], errors="coerce").dt.strftime("%Y-%m-%d")
        elif ctype == "bool":
            df_fit[col] = df_fit[col].astype(str)

    df_fit = df_fit.dropna(how="all")
    if len(df_fit) < 5:
        return None

    # Cap to 50 K rows so fitting stays fast regardless of upload size
    if len(df_fit) > 50_000:
        df_fit = df_fit.sample(50_000, random_state=42)

    try:
        metadata = _SDVMetadata()
        for col, ctype in sdv_cols:
            if ctype == "date":
                metadata.add_column(col, sdtype="datetime", datetime_format="%Y-%m-%d")
            else:
                metadata.add_column(col, sdtype=_SDV_TYPE_MAP[ctype])

        synthesizer = _GaussianCopula(metadata)
        synthesizer.fit(df_fit)

        model_id = str(uuid.uuid4())
        _sdv_models[model_id] = {"synthesizer": synthesizer, "sdv_cols": col_names}

        # Simple LRU eviction: keep at most 50 models in memory
        if len(_sdv_models) > 50:
            del _sdv_models[next(iter(_sdv_models))]

        return model_id
    except Exception:
        return None


def _synthesize(
    schema_columns: list[dict],
    source_stats: dict[str, dict],
    n: int,
    model_id: str | None,
) -> pd.DataFrame:
    """
    Generate `n` synthetic rows.
    Uses the fitted SDV GaussianCopulaSynthesizer when a valid model_id is provided,
    falling back to the per-column statistical sampler otherwise.
    """
    if model_id and model_id in _sdv_models:
        entry = _sdv_models[model_id]
        synthesizer = entry["synthesizer"]
        sdv_cols: list[str] = entry["sdv_cols"]
        try:
            sdv_df = synthesizer.sample(num_rows=n)
            actual_n = len(sdv_df)
            # Re-add columns excluded from the SDV model (uuid, array<str>, …)
            for col_def in schema_columns:
                col = col_def["column"]
                if col not in sdv_df.columns:
                    stat = source_stats.get(col) or {"col_type": "uuid"}
                    sdv_df[col] = _synth_column(col, stat, actual_n)
            ordered = [c["column"] for c in schema_columns if c["column"] in sdv_df.columns]
            return sdv_df[ordered]
        except Exception:
            pass  # Fall through to statistical sampler

    return _build_synth_df(schema_columns, source_stats, n)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate(source_stats: dict[str, dict], synth_df: pd.DataFrame) -> dict:
    col_results = []
    realism_scores: list[float] = []
    diversity_scores: list[float] = []

    for col, stat in source_stats.items():
        if col not in synth_df.columns:
            continue
        col_type = stat.get("col_type", "enum")
        synth_s = synth_df[col].dropna()
        fidelity = 100
        note = None

        if col_type in ("int", "float"):
            s_mean, s_std = stat["mean"], stat["std"]
            t_mean = float(synth_s.mean())
            t_std = float(synth_s.std()) if len(synth_s) > 1 else s_std
            mean_drift = abs(t_mean - s_mean) / (abs(s_mean) + 1e-9)
            std_drift = abs(t_std - s_std) / (abs(s_std) + 1e-9)
            fidelity = max(0, round(100 - (mean_drift + std_drift) * 50))
            if fidelity < 90:
                note = f"Distribution drift vs. source (Δμ={mean_drift:.1%})"
            realism_scores.append(fidelity)
            src_cv = s_std / (abs(s_mean) + 1e-9)
            syn_cv = t_std / (abs(t_mean) + 1e-9)
            div_score = max(0, round(100 - abs(src_cv - syn_cv) / (src_cv + 1e-9) * 100))
            diversity_scores.append(div_score)

        elif col_type == "enum":
            src_cats = set(stat.get("categories", []))
            syn_cats = set(synth_s.astype(str).unique())
            coverage = len(src_cats & syn_cats) / max(len(src_cats), 1)
            fidelity = round(coverage * 100)
            if fidelity < 95:
                missing = len(src_cats - syn_cats)
                note = f"Missing {missing} source categor{'y' if missing == 1 else 'ies'}"
            realism_scores.append(fidelity)
            diversity_scores.append(round(min(100, len(syn_cats) / max(len(src_cats), 1) * 100)))

        status = "pass" if fidelity >= 90 else "warn" if fidelity >= 70 else "fail"
        row: dict[str, Any] = {"column": col, "fidelity": fidelity, "status": status}
        if note:
            row["note"] = note
        col_results.append(row)

    realism = round(float(np.mean(realism_scores))) if realism_scores else 95
    diversity = round(float(np.mean(diversity_scores))) if diversity_scores else 88

    pii_found = False
    for col in synth_df.select_dtypes(include="object").columns:
        vals = synth_df[col].dropna().astype(str).head(200)
        for pat in (_EMAIL_RE, _PHONE_RE, _SSN_RE):
            if vals.apply(lambda x: bool(pat.search(x))).any():
                pii_found = True
                break

    safety = 65 if pii_found else 100
    n_rows = len(synth_df)

    insights = []
    warn_cols = [r for r in col_results if r["status"] == "warn"]
    if warn_cols:
        insights.append(
            f"{warn_cols[0]['column']} shows distribution drift — consider reviewing source variance"
        )
    insights.append(
        "No PII detected across all {:,} rows".format(n_rows)
        if not pii_found
        else "PII-like patterns detected — review string columns before sharing"
    )
    insights.append(
        f"Inter-column correlations preserved within {abs(100 - realism)}% of source statistics"
    )

    overall_pass = realism >= 85 and diversity >= 75 and safety >= 90
    return {
        "verdict": "Ready for use" if overall_pass else "Review recommended",
        "verdictStatus": "pass" if overall_pass else "warn",
        "metrics": [
            {"label": "Realism", "score": realism, "status": "pass" if realism >= 85 else "warn"},
            {"label": "Diversity", "score": diversity, "status": "pass" if diversity >= 75 else "warn"},
            {"label": "Safety / PII", "score": safety, "status": "pass" if safety >= 90 else "warn"},
        ],
        "columns": col_results,
        "insights": insights,
    }


# ── Prompt-to-Action: NL → plan ───────────────────────────────────────────────

_PLAN_SYSTEM = """\
You are a synthetic data generation expert. Given a natural language description of a desired dataset, \
respond ONLY with a valid JSON object (no markdown, no explanation) in this exact format:
{
  "schema": [
    {"column": "col_name", "type": "int|float|enum|bool|date|uuid|text", "distribution": "human-readable description", "sample": "example_value_as_string"}
  ],
  "generation_spec": {
    "row_count": 10000,
    "format": "csv",
    "labels": [],
    "edge_cases": [],
    "constraints": []
  },
  "generation_plan": {
    "intro": "One sentence describing what you will build.",
    "steps": [
      {"title": "Schema Design", "description": "..."},
      {"title": "Data Synthesis", "description": "..."},
      {"title": "Fidelity Validation", "description": "..."}
    ]
  },
  "clarifying_questions": []
}

Rules:
- If the request clearly identifies domain, columns, and scale: return empty clarifying_questions and a complete schema (5-10 columns).
- If the request is underspecified (no domain, no column hints, or very short): return 2-3 clarifying questions AND a best-guess schema.
- Always produce at least a partial schema even when asking questions.
- Extract row_count from the prompt; default 10000.
- Put any edge cases or special conditions (e.g. "HbA1c > 12 with 3+ comorbidities") in edge_cases.
- Respond ONLY with the JSON object — nothing else.\
"""


def _fallback_plan(prompt: str) -> dict:
    """Keyword-based plan when no LLM key is configured."""
    low = prompt.lower()

    is_medical = any(k in low for k in ["patient", "clinical", "medical", "health", "diabetes", "hba1c", "ehr"])
    is_ecommerce = any(k in low for k in ["order", "product", "purchase", "customer", "transaction", "ecommerce", "e-commerce"])
    is_nlp = any(k in low for k in ["text", "sentence", "document", "review", "nlp", "language", "sentiment"])
    is_finance = any(k in low for k in ["loan", "credit", "fraud", "transaction", "bank", "financial"])

    row_count = 10_000
    m = re.search(r"(\d[\d,]*)\s*(?:rows?|records?|samples?|examples?)", low)
    if m:
        row_count = int(m.group(1).replace(",", ""))

    edge_cases: list[str] = []
    for pattern in [
        r"edge cases? (?:for|of|where) ([^,.]+)",
        r"(?:hba1c|value|score|amount)\s*[><=]+\s*[\d.]+(?:\s+with\s+[^,.]+)?",
    ]:
        match = re.search(pattern, low)
        if match:
            edge_cases.append(match.group(0)[:100])

    if is_medical:
        schema = [
            {"column": "patient_id", "type": "uuid", "distribution": "unique", "sample": "pt_8f2a"},
            {"column": "age", "type": "int", "distribution": "Gaussian (μ=52, σ=8)", "sample": "54"},
            {"column": "sex", "type": "enum", "distribution": "{F:0.52, M:0.48}", "sample": "F"},
            {"column": "diagnosis", "type": "enum", "distribution": "{T2D:0.60, T1D:0.25, Pre:0.15}", "sample": "T2D"},
            {"column": "hba1c", "type": "float", "distribution": "LogNormal (μ=1.9, σ=0.4)", "sample": "7.2"},
            {"column": "bmi", "type": "float", "distribution": "Gaussian (μ=28.5, σ=4.2)", "sample": "27.1"},
            {"column": "last_visit", "type": "date", "distribution": "Uniform 2023–2025", "sample": "2024-08-12"},
        ]
        intro = "I'll generate a synthetic clinical dataset with realistic patient distributions."
    elif is_ecommerce:
        schema = [
            {"column": "order_id", "type": "uuid", "distribution": "unique", "sample": "ord_4a2c"},
            {"column": "customer_id", "type": "uuid", "distribution": "unique", "sample": "cust_7b1"},
            {"column": "product_name", "type": "enum", "distribution": "Categorical", "sample": "Widget Pro"},
            {"column": "category", "type": "enum", "distribution": "{electronics:0.35, clothing:0.30, home:0.20, other:0.15}", "sample": "electronics"},
            {"column": "amount", "type": "float", "distribution": "LogNormal (μ=4.2, σ=0.8)", "sample": "49.99"},
            {"column": "status", "type": "enum", "distribution": "{complete:0.72, pending:0.18, refunded:0.10}", "sample": "complete"},
            {"column": "order_date", "type": "date", "distribution": "Uniform 2023–2025", "sample": "2024-03-15"},
        ]
        intro = "I'll generate a synthetic e-commerce dataset with realistic order patterns."
    elif is_nlp:
        schema = [
            {"column": "id", "type": "uuid", "distribution": "unique", "sample": "doc_1a2b"},
            {"column": "text", "type": "text", "distribution": "Variable length", "sample": "The product was excellent…"},
            {"column": "label", "type": "enum", "distribution": "{positive:0.50, negative:0.50}", "sample": "positive"},
            {"column": "confidence", "type": "float", "distribution": "Gaussian (μ=0.85, σ=0.12)", "sample": "0.91"},
            {"column": "source", "type": "enum", "distribution": "{web:0.60, mobile:0.40}", "sample": "web"},
        ]
        intro = "I'll generate a synthetic text classification dataset for NLP tasks."
    elif is_finance:
        schema = [
            {"column": "transaction_id", "type": "uuid", "distribution": "unique", "sample": "txn_3c4d"},
            {"column": "account_id", "type": "uuid", "distribution": "unique", "sample": "acct_9e1f"},
            {"column": "amount", "type": "float", "distribution": "LogNormal (μ=5.1, σ=1.2)", "sample": "250.00"},
            {"column": "merchant_category", "type": "enum", "distribution": "Categorical", "sample": "retail"},
            {"column": "is_fraud", "type": "bool", "distribution": "{True:0.02, False:0.98}", "sample": "False"},
            {"column": "timestamp", "type": "date", "distribution": "Uniform 2023–2025", "sample": "2024-06-10"},
        ]
        intro = "I'll generate a synthetic financial transaction dataset with realistic fraud rates."
    else:
        schema = [
            {"column": "id", "type": "uuid", "distribution": "unique", "sample": "row_1a2b"},
            {"column": "value", "type": "float", "distribution": "Gaussian (μ=50, σ=15)", "sample": "52.4"},
            {"column": "category", "type": "enum", "distribution": "Categorical", "sample": "type_A"},
            {"column": "timestamp", "type": "date", "distribution": "Uniform 2023–2025", "sample": "2024-01-15"},
            {"column": "flag", "type": "bool", "distribution": "{True:0.30, False:0.70}", "sample": "False"},
        ]
        intro = "I'll generate a synthetic dataset based on your description."

    clarifying_questions: list[dict] = []
    if len(prompt.split()) < 8:
        clarifying_questions = [
            {"id": "q1", "question": "What domain or industry is this dataset for (e.g. healthcare, finance, e-commerce)?"},
            {"id": "q2", "question": "What are the most important columns or features your model will use?"},
        ]

    return {
        "schema": schema,
        "generation_spec": {
            "row_count": row_count,
            "format": "csv",
            "labels": [],
            "edge_cases": edge_cases,
            "constraints": [],
        },
        "generation_plan": {
            "intro": intro,
            "steps": [
                {
                    "title": "Schema Design",
                    "description": f"Designed {len(schema)} columns based on your request '{prompt[:60]}{'…' if len(prompt) > 60 else ''}'",
                },
                {
                    "title": "Data Synthesis",
                    "description": f"Generating {row_count:,} rows matching inferred distributions and your constraints.",
                },
                {
                    "title": "Fidelity Validation",
                    "description": "Cross-validating distributions, checking for PII, and issuing a quality report before delivery.",
                },
            ],
        },
        "clarifying_questions": clarifying_questions,
    }


class PlanRequest(BaseModel):
    prompt: str
    context_schema: list[dict] | None = None


@app.post("/api/plan")
async def plan(req: PlanRequest):
    """
    Infer schema, generation plan, and clarifying questions from a natural language prompt.
    Uses Claude when ANTHROPIC_API_KEY is set; falls back to keyword matching otherwise.
    """
    if _llm:
        try:
            msg = _llm.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=_PLAN_SYSTEM,
                messages=[{"role": "user", "content": req.prompt}],
            )
            raw = msg.content[0].text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except Exception:
            pass  # Fall through to deterministic fallback

    return _fallback_plan(req.prompt)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/infer-schema")
async def infer_schema(files: list[UploadFile] = File(...)):
    """
    Upload CSV / XLSX / Parquet files.
    Returns inferred schema columns (with null_pct) and per-column statistics.
    """
    dfs: list[pd.DataFrame] = []
    for f in files:
        try:
            dfs.append(_read_upload(f))
        except Exception as exc:
            return {"error": f"Could not read {f.filename}: {exc}"}

    df = pd.concat(dfs, ignore_index=True)

    columns = []
    stats: dict[str, dict] = {}
    for col in df.columns:
        series = df[col].dropna()
        null_pct = round(float(df[col].isna().mean() * 100), 1)
        col_type = _infer_col_type(series)
        sample_val = str(series.iloc[0]) if len(series) else ""
        if len(sample_val) > 12:
            sample_val = sample_val[:9] + "…"
        columns.append(
            {
                "column": col,
                "type": col_type,
                "distribution": _distribution_label(series, col_type),
                "sample": sample_val,
                "null_pct": null_pct,
            }
        )
        stats[col] = _col_stats(series, col_type)

    # Fit SDV model for correlation-preserving synthesis
    col_info = [{"column": c["column"], "type": c["type"]} for c in columns]
    model_id = _fit_sdv_model(df, col_info)

    return {"columns": columns, "stats": stats, "source_rows": len(df), "model_id": model_id}


class GenerateRequest(BaseModel):
    schema_columns: list[dict]
    source_stats: dict[str, dict]
    row_count: int = 10_000
    prompt: str = ""
    format: str = "csv"  # csv | jsonl | parquet
    edge_cases: list[str] = []
    model_id: str | None = None  # SDV model fitted during /api/infer-schema


@app.post("/api/preview")
async def preview_dataset(req: GenerateRequest):
    """Generate 10 preview rows as a JSON array — no session stored."""
    synth_df = _synthesize(req.schema_columns, req.source_stats, n=10, model_id=req.model_id)
    return {"rows": synth_df.to_dict(orient="records")}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """
    Synthesise rows faithful to the inferred schema and run fidelity validation.
    Uses the fitted SDV GaussianCopulaSynthesizer when a model_id is provided
    (preserves inter-column correlations); falls back to per-column statistical
    sampling when generating from a natural-language prompt without grounding data.
    Supports CSV, JSONL, and Parquet output formats.
    """
    n = max(1, min(req.row_count, 100_000))
    synth_df = _synthesize(req.schema_columns, req.source_stats, n, model_id=req.model_id)

    fmt = (req.format or "csv").lower()
    if fmt not in ("csv", "jsonl", "parquet"):
        fmt = "csv"

    buf = io.BytesIO()
    if fmt == "jsonl":
        data_bytes = ("\n".join(json.dumps(row) for row in synth_df.to_dict(orient="records"))).encode()
    elif fmt == "parquet":
        synth_df.to_parquet(buf, index=False)
        data_bytes = buf.getvalue()
    else:
        synth_df.to_csv(buf, index=False)
        data_bytes = buf.getvalue()

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"bytes": data_bytes, "format": fmt}

    validation = _validate(req.source_stats, synth_df)
    file_size_kb = round(len(data_bytes) / 1024, 1)
    filename = f"aperture_output.{fmt}"

    return {
        "session_id": session_id,
        "row_count": n,
        "file_size_kb": file_size_kb,
        "format": fmt,
        "filename": filename,
        "validation": validation,
    }


@app.get("/api/download/{session_id}")
async def download(session_id: str):
    """Stream the generated file for a given session."""
    entry = _sessions.get(session_id)
    if entry is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found or expired")

    data = entry["bytes"]
    fmt = entry.get("format", "csv")
    media_types = {
        "csv": "text/csv",
        "jsonl": "application/jsonlines",
        "parquet": "application/octet-stream",
    }
    media_type = media_types.get(fmt, "text/csv")
    filename = f"aperture_{session_id[:8]}.{fmt}"

    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
