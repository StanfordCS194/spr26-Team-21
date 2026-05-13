"""POST /api/sources/mongo/* — connect to a user-supplied MongoDB cluster."""
from fastapi import APIRouter
from pydantic import BaseModel

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
