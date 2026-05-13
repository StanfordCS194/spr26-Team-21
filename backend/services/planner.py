"""Natural language → schema plan: LLM-backed with deterministic keyword fallback."""
import re

PLAN_SYSTEM = """\
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


def fallback_plan(prompt: str) -> dict:
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
