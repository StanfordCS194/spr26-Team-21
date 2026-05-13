"""Pydantic request models."""
from pydantic import BaseModel


class PlanRequest(BaseModel):
    prompt: str
    context_schema: list[dict] | None = None


class GenerateRequest(BaseModel):
    schema_columns: list[dict]
    source_stats: dict[str, dict]
    row_count: int = 10_000
    prompt: str = ""
    format: str = "csv"  # csv | jsonl | parquet
    edge_cases: list[str] = []
    model_id: str | None = None  # SDV model fitted during /api/infer-schema
