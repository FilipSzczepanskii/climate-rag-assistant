"""Gradio chat UI for the RAG assistant."""

from __future__ import annotations

import logging

import gradio as gr

from app.observability import metrics
from app.rag import RagAnswer, RagPipeline

logger = logging.getLogger("rag.ui")

_EXAMPLES = [
    "What are the main health effects of PM2.5?",
    "How does the greenhouse effect work?",
    "What does the Paris Agreement commit countries to?",
    "What causes acid rain?",
]


def _format_answer(result: RagAnswer) -> str:
    parts = [result.answer]
    if result.sources:
        parts.append("\n---\n**Sources**")
        parts.extend(f"- [{s.title}]({s.source_url})" for s in result.sources)
    return "\n".join(parts)


def build_ui(pipeline: RagPipeline) -> gr.Blocks:
    """Build the Gradio chat interface backed by the RAG pipeline."""

    def respond(message: str, _history: list) -> str:
        try:
            result = pipeline.answer(message)
        except Exception as exc:  # noqa: BLE001 - never crash the chat surface
            metrics.record_error()
            logger.exception("chat query failed")
            return f"Something went wrong while answering: {exc}"
        metrics.record_query(result.latency_ms)
        return _format_answer(result)

    with gr.Blocks(title="Climate Q&A", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Climate Q&A\n"
            "Ask about climate, air quality and environmental science. "
            "Every answer is retrieved from a Wikipedia knowledge base and cites "
            "the articles it draws on."
        )
        gr.ChatInterface(
            fn=respond,
            examples=_EXAMPLES,
            fill_height=True,
        )
    return demo
