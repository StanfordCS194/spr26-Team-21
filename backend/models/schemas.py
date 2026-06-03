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
    source_id: str | None = None  # Source dataframe stored during /api/infer-schema
    label_col: str | None = None  # Target column for the Utility (TSTR) pillar; auto-detected when None
    discovered_suggestions: list[dict] = []  # From /api/discover-edge-cases; surfaced in Trust Report


class DiscoverEdgeCasesRequest(BaseModel):
    schema_columns: list[dict]
    source_stats: dict[str, dict]
    source_id: str | None = None  # If provided, conjunction detection uses the retained source DataFrame