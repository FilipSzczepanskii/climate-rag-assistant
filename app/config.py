"""Central configuration.

All knobs are read from environment variables (or a local .env file) so the
same code runs unchanged in local dev, CI, and the Hugging Face Space.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Retrieval / embeddings ------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800          # target characters per chunk
    chunk_overlap: int = 120       # characters carried between adjacent chunks
    top_k: int = 4                 # chunks passed to the LLM as context

    # --- Reranking (optional) --------------------------------------------
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_candidates: int = 20    # retrieved before the reranker narrows to top_k

    # --- LLM -------------------------------------------------------------
    llm_provider: str = "huggingface"   # "huggingface" | "groq"
    llm_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    hf_token: str | None = None
    groq_api_key: str | None = None

    # --- Paths -----------------------------------------------------------
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus.parquet"
    chroma_path: Path = PROJECT_ROOT / "data" / "chroma"
    collection_name: str = "climate_docs"

    # --- App -------------------------------------------------------------
    app_title: str = "Climate Q&A"
    log_level: str = "INFO"

    def resolved_llm_model(self) -> str:
        """Pick a sensible default model for the active provider."""
        if self.llm_provider == "groq" and self.llm_model.startswith("meta-llama/"):
            # The Groq catalogue uses short model names.
            return "llama-3.3-70b-versatile"
        return self.llm_model


settings = Settings()
