"""POST /api/sources/mongo/* — connect to a user-supplied MongoDB cluster."""
import json
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.agents.sourcing_agent import run_agent
from services.profile import profile_dataframe
from services.sources import mongo

router = APIRouter(prefix="/api/sources/mongo")


class TestBody(BaseModel):
    uri: str


class CollectionsBody(BaseModel):
    uri: str
    db: str


class InferBody(BaseModel):
    uri: str
    db: str
    collection: str


class AutoInferBody(BaseModel):
    uri: str
    db: str
    collection: Optional[str] = None
    prompt: str


@router.post("/test")
def test_connection(body: TestBody):
    return mongo.list_databases(body.uri)


@router.post("/collections")
def collections(body: CollectionsBody):
    return mongo.list_collections(body.uri, body.db)


@router.post("/infer-schema")
def infer_from_mongo(body: InferBody):
    try:
        df = mongo.read_collection(body.uri, body.db, body.collection)
    except Exception as exc:
        return {"error": f"Could not read {body.db}.{body.collection}: {exc}"}

    if df.empty:
        return {"error": f"{body.db}.{body.collection} returned no documents"}

    payload = profile_dataframe(df)
    payload["host"] = mongo.safe_host(body.uri)
    return payload


@router.post("/auto-infer")
def auto_infer(body: AutoInferBody):
    """Run the sourcing agent against the connected cluster, streamed as SSE."""

    def event_stream():
        try:
            for event in run_agent(
                uri=body.uri,
                db=body.db,
                collection=body.collection,
                prompt=body.prompt,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
