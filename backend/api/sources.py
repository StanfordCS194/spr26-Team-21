"""POST /api/sources/* — connect to a user-supplied MongoDB cluster or S3 bucket."""
import json
from typing import Any, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.agents.sourcing_agent import run_mongo_agent, run_s3_agent
from services.profile import profile_dataframe
from services.sources import mongo, s3

router = APIRouter(prefix="/api/sources/mongo")
s3_router = APIRouter(prefix="/api/sources/s3")


# ── MongoDB ──────────────────────────────────────────────────────────────────


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
            for event in run_mongo_agent(
                uri=body.uri,
                db=body.db,
                collection=body.collection,
                prompt=body.prompt,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Amazon S3 ────────────────────────────────────────────────────────────────


class S3Creds(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump()


class S3TestBody(BaseModel):
    creds: S3Creds


class S3ObjectsBody(BaseModel):
    creds: S3Creds
    bucket: str
    prefix: Optional[str] = ""


class S3InferBody(BaseModel):
    creds: S3Creds
    bucket: str
    key: str


class S3AutoInferBody(BaseModel):
    creds: S3Creds
    bucket: str
    prefix: Optional[str] = None
    key: Optional[str] = None
    prompt: str


@s3_router.post("/test")
def s3_test(body: S3TestBody):
    return s3.list_buckets(body.creds.as_dict())


@s3_router.post("/objects")
def s3_objects(body: S3ObjectsBody):
    return s3.list_objects(body.creds.as_dict(), body.bucket, body.prefix or "")


@s3_router.post("/infer-schema")
def infer_from_s3(body: S3InferBody):
    try:
        df = s3.read_object(body.creds.as_dict(), body.bucket, body.key)
    except Exception as exc:
        return {"error": f"Could not read s3://{body.bucket}/{body.key}: {exc}"}

    if df.empty:
        return {"error": f"s3://{body.bucket}/{body.key} returned no rows"}

    payload = profile_dataframe(df)
    payload["host"] = s3.safe_host(body.bucket)
    return payload


@s3_router.post("/auto-infer")
def s3_auto_infer(body: S3AutoInferBody):
    """Run the sourcing agent against the connected bucket, streamed as SSE."""

    def event_stream():
        try:
            for event in run_s3_agent(
                creds=body.creds.as_dict(),
                bucket=body.bucket,
                prefix=body.prefix,
                key=body.key,
                prompt=body.prompt,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
