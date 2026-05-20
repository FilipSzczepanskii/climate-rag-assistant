"""Provider-agnostic chat LLM client.

Hugging Face Inference and Groq both expose OpenAI-compatible chat endpoints,
so one OpenAI client pointed at the right base URL covers both providers. The
active provider is chosen by configuration and call sites never branch on it.
"""

from __future__ import annotations

from openai import OpenAI

from app.config import Settings

_BASE_URLS = {
    "huggingface": "https://router.huggingface.co/v1",
    "groq": "https://api.groq.com/openai/v1",
}


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot return a completion."""


class LLMClient:
    """Wraps an OpenAI-compatible chat provider behind a single complete()."""

    def __init__(self, settings: Settings) -> None:
        self.provider = settings.llm_provider
        self.model = settings.resolved_llm_model()
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens

        if self.provider not in _BASE_URLS:
            raise LLMError(f"Unknown LLM provider: {self.provider}")

        api_key = (
            settings.hf_token
            if self.provider == "huggingface"
            else settings.groq_api_key
        )
        if not api_key:
            env_var = "HF_TOKEN" if self.provider == "huggingface" else "GROQ_API_KEY"
            raise LLMError(f"{env_var} is not set for provider '{self.provider}'")

        self._client = OpenAI(base_url=_BASE_URLS[self.provider], api_key=api_key)

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - surface any provider error uniformly
            raise LLMError(f"{self.provider} completion failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned an empty completion")
        return content.strip()
