"""Aperture backend — statistical data synthesis pipeline.

Run:  uvicorn main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import generate, health, plan, schema

app = FastAPI(title="Aperture API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(plan.router)
app.include_router(schema.router)
app.include_router(generate.router)
