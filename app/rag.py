"""The retrieval-augmented generation pipeline.

Ties retrieval and generation together: retrieve passages, assemble a prompt
from a versioned template, call the LLM, and return the answer alongside the
sources it was grounded in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.embeddings import Embedder
from app.llm import LLMClient
from app.retriever import Retriever
from app.vectorstore import Hit, VectorStore

_PROMPT_PATH = Path(__file__).parent / "prompts" / "rag_v1.txt"


@dataclass
class Source:
    title: str
    source_url: str
    score: float


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    latency_ms: int = 0
    retrieved: int = 0


def _format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks into the labelled blocks the prompt expects."""
    return "\n\n---\n\n".join(f"[{hit.title}]\n{hit.text}" for hit in hits)


def _unique_sources(hits: list[Hit]) -> list[Source]:
    """Collapse chunks to one entry per article, keeping the best score."""
    best: dict[str, Source] = {}
    for hit in hits:
        current = best.get(hit.title)
        if current is None or hit.score > current.score:
            best[hit.title] = Source(hit.title, hit.source_url, hit.score)
    return sorted(best.values(), key=lambda s: s.score, reverse=True)


class RagPipeline:
    """End-to-end question answering over the indexed corpus."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._prompt_template = _PROMPT_PATH.read_text(encoding="utf-8")

        embedder = Embedder(settings.embedding_model)
        store = VectorStore(settings.chroma_path, settings.collection_name)
        self._ensure_index(store, settings)

        self.retriever = Retriever(
            store,
            embedder,
            top_k=settings.top_k,
            use_reranker=settings.use_reranker,
            reranker_model=settings.reranker_model,
            rerank_candidates=settings.rerank_candidates,
        )
        self._llm: LLMClient | None = None

    @staticmethod
    def _ensure_index(store: VectorStore, settings: Settings) -> None:
        """Populate the vector store from the committed corpus parquet if empty."""
        if store.count() > 0:
            return
        if not settings.corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus not found at {settings.corpus_path}. "
                "Run: python -m ingest.build_index"
            )
        store.build_from_parquet(settings.corpus_path)

    @property
    def llm(self) -> LLMClient:
        # Built lazily so retrieval-only paths and tests need no API key.
        if self._llm is None:
            self._llm = LLMClient(self.settings)
        return self._llm

    def answer(self, question: str) -> RagAnswer:
        started = time.perf_counter()
        question = question.strip()
        if not question:
            return RagAnswer(question=question, answer="Please ask a question.")

        hits = self.retriever.retrieve(question)
        if not hits:
            return RagAnswer(
                question=question,
                answer="I do not have enough information to answer that.",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        prompt = self._prompt_template.format(
            context=_format_context(hits),
            question=question,
        )
        answer_text = self.llm.complete(prompt)

        return RagAnswer(
            question=question,
            answer=answer_text,
            sources=_unique_sources(hits),
            latency_ms=int((time.perf_counter() - started) * 1000),
            retrieved=len(hits),
        )
