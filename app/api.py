"""FastAPI service for the RAG assistant.

Exposes a JSON query API and mounts the Gradio chat UI at the root path. One
process serves both, which is exactly what the Hugging Face Space runs.

Run:
    python -m app.api
"""

from __future__ import annotations

import logging

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.observability import configure_logging, metrics
from app.rag import RagPipeline
from app.ui import build_ui

configure_logging(settings.log_level)
logger = logging.getLogger("rag.api")

pipeline = RagPipeline(settings)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class SourceModel(BaseModel):
    title: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    latency_ms: int
    retrieved: int


app = FastAPI(title=settings.app_title, version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": pipeline.retriever.store.count(),
        "llm_provider": settings.llm_provider,
        "metrics": metrics.snapshot(),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        result = pipeline.answer(request.question)
    except Exception as exc:  # noqa: BLE001 - map any failure to a 502
        metrics.record_error()
        logger.exception("query failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    metrics.record_query(result.latency_ms)
    logger.info(
        "answered in %d ms over %d chunks", result.latency_ms, result.retrieved
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceModel(**vars(s)) for s in result.sources],
        latency_ms=result.latency_ms,
        retrieved=result.retrieved,
    )


# Mount the Gradio chat UI at the root path.
app = gr.mount_gradio_app(app, build_ui(pipeline), path="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
