"""LLM clients with model-availability handling, retries, and token/latency metadata.

Replaces src/models/llm.py. Keeps `GeminiLLM().generate(prompt)` working, and adds:

  * generate_with_meta(prompt) -> LLMResult (text, latency, tokens, model_used, error)
  * a live check of which model names your API key can actually use, with fallback
  * retry/backoff for rate limits and transient 5xx errors
  * an OpenAI-compatible client (GPT-4.1 mini via OpenAI, Qwen3-8B via Ollama/vLLM)
  * get_llm("provider:model") factory so experiments can loop over models

Env vars:
  GEMINI_API_KEY, GEMINI_FALLBACK_MODELS (comma list, default "gemini-2.5-flash")
  OPENAI_API_KEY, OPENAI_BASE_URL (default https://api.openai.com/v1)
  OLLAMA_BASE_URL (default http://localhost:11434/v1)
"""

import os
import re
import time
from dataclasses import asdict, dataclass

from src.config import GEMINI_API_KEY, GEMINI_MODEL

RETRYABLE_MARKERS = (
    "429", "500", "502", "503", "504",
    "resource_exhausted", "unavailable", "overloaded",
    "rate limit", "timeout", "timed out", "temporarily",
)
MODEL_MISSING_MARKERS = (
    "404", "not_found", "not found", "is not supported", "does not exist",
    "no longer available",
)


@dataclass
class LLMResult:
    text: str = ""
    model_requested: str = ""
    model_used: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int = None
    completion_tokens: int = None
    total_tokens: int = None
    error: str = None
    fallback_used: bool = False

    def to_dict(self):
        return asdict(self)


def _looks_like(markers, exc):
    msg = str(exc).lower()
    return any(m in msg for m in markers)


class BaseLLM:
    name = "base"

    def generate_with_meta(self, prompt):
        raise NotImplementedError

    def generate(self, prompt):
        result = self.generate_with_meta(prompt)
        if result.error:
            raise RuntimeError(result.error)
        return result.text


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #
class GeminiLLM(BaseLLM):
    def __init__(self, model=None, temperature=0.0, max_retries=4):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")

        from google import genai  # imported lazily so other providers work without it

        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.temperature = temperature
        self.max_retries = max_retries

        requested = model or GEMINI_MODEL
        fallbacks = [
            m.strip()
            for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.5-flash").split(",")
            if m.strip()
        ]
        self.requested = requested
        self.candidates = [requested] + [m for m in fallbacks if m != requested]
        self.name = f"gemini:{requested}"

        self.available = self._list_available_models()
        self.model = self._pick_model()

    def _list_available_models(self):
        """Names your key can use, without the 'models/' prefix. None if listing fails."""
        try:
            return {m.name.replace("models/", "") for m in self.client.models.list()}
        except Exception as exc:  # listing is best-effort
            print(f"[llm] could not list Gemini models ({exc}); will rely on fallbacks.")
            return None

    def _pick_model(self):
        if self.available is None:
            return self.candidates[0]
        for candidate in self.candidates:
            if candidate in self.available:
                if candidate != self.requested:
                    print(
                        f"[llm] '{self.requested}' is not available for this key; "
                        f"using '{candidate}' instead."
                    )
                return candidate
        flash = sorted(m for m in self.available if "flash" in m)[:15]
        raise RuntimeError(
            f"None of {self.candidates} are available for this API key. "
            f"Some available flash models: {flash}. "
            "Set GEMINI_MODEL / GEMINI_FALLBACK_MODELS to one of those."
        )

    def generate_with_meta(self, prompt):
        from google.genai import types

        result = LLMResult(model_requested=self.requested)
        order = [self.model] + [m for m in self.candidates if m != self.model]
        start = time.perf_counter()
        last_error = None

        for model in order:
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=self.temperature),
                    )
                    usage = getattr(response, "usage_metadata", None)
                    result.text = response.text or ""
                    result.model_used = model
                    result.fallback_used = model != self.requested
                    result.prompt_tokens = getattr(usage, "prompt_token_count", None)
                    result.completion_tokens = getattr(usage, "candidates_token_count", None)
                    result.total_tokens = getattr(usage, "total_token_count", None)
                    result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                    return result
                except Exception as exc:
                    last_error = exc
                    if _looks_like(MODEL_MISSING_MARKERS, exc):
                        break  # try the next candidate model
                    if _looks_like(RETRYABLE_MARKERS, exc) and attempt < self.max_retries:
                        time.sleep(min(2 ** attempt * 2, 30))
                        continue
                    break

        result.error = f"{type(last_error).__name__}: {last_error}"
        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return result


# --------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Ollama, vLLM, LM Studio ...)
# --------------------------------------------------------------------------- #
class OpenAICompatLLM(BaseLLM):
    def __init__(self, model, base_url=None, api_key=None, temperature=0.0,
                 max_retries=4, timeout=300, label="openai"):
        import requests  # already in requirements.txt

        self._requests = requests
        self.model = model
        self.requested = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.name = f"{label}:{model}"

    def generate_with_meta(self, prompt):
        result = LLMResult(model_requested=self.requested)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers, json=payload, timeout=self.timeout,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                text = data["choices"][0]["message"]["content"] or ""
                # Qwen3 / reasoning models may emit <think>...</think> blocks.
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                usage = data.get("usage") or {}
                result.text = text
                result.model_used = data.get("model", self.model)
                result.prompt_tokens = usage.get("prompt_tokens")
                result.completion_tokens = usage.get("completion_tokens")
                result.total_tokens = usage.get("total_tokens")
                result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
                return result
            except Exception as exc:
                last_error = exc
                if _looks_like(RETRYABLE_MARKERS, exc) and attempt < self.max_retries:
                    time.sleep(min(2 ** attempt * 2, 30))
                    continue
                break

        result.error = f"{type(last_error).__name__}: {last_error}"
        result.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return result


# --------------------------------------------------------------------------- #
# Test double
# --------------------------------------------------------------------------- #
class FakeLLM(BaseLLM):
    """Deterministic stand-in for tests / dry runs. `responder(prompt) -> str`."""

    def __init__(self, responder=None, name="fake:fake"):
        self.responder = responder or (lambda prompt: "Reasoning: n/a\nFinal Answer: INSUFFICIENT_EVIDENCE")
        self.name = name
        self.requested = name.split(":", 1)[-1]

    def generate_with_meta(self, prompt):
        start = time.perf_counter()
        text = self.responder(prompt)
        return LLMResult(
            text=text,
            model_requested=self.requested,
            model_used=self.requested,
            latency_ms=round((time.perf_counter() - start) * 1000, 3),
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
        )


def get_llm(spec):
    """'gemini:gemini-2.5-flash' | 'openai:gpt-4.1-mini' | 'ollama:qwen3:8b' | 'fake'."""
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider == "gemini":
        return GeminiLLM(model=model or None)
    if provider == "openai":
        return OpenAICompatLLM(model or "gpt-4.1-mini", label="openai")
    if provider == "ollama":
        return OpenAICompatLLM(
            model or "qwen3:8b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama", label="ollama",
        )
    if provider == "fake":
        return FakeLLM()
    raise ValueError(f"Unknown LLM spec '{spec}'. Use provider:model, e.g. gemini:gemini-2.5-flash")
