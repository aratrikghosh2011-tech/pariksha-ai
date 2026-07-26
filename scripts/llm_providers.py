"""
llm_providers.py

Provider-agnostic interface for the generation step of the RAG pipeline.
Swap backends by changing PROVIDER in your .env, no other code changes.

Requires a .env file in the repo root with:
  GEMINI_API_KEY=your_key_here
  NVIDIA_API_KEY=your_key_here      # from build.nvidia.com, for Nemotron 3
  PROVIDER=gemini                   # or: nemotron

--- Free-tier model choice (as of July 26, 2026) ---

This has been wrong twice already, so here's the actual history so
nobody re-guesses it a third time:

  - gemini-3.5-flash: 20 req/day free. This is what kept running out.
  - gemini-2.5-flash-lite: WAS believed to be 1,000/day based on
    published docs, but Aratrik's OWN account dashboard
    (aistudio.google.com/rate-limit) showed its real RPD limit is also
    just 20 on this account, right now. Published numbers and your
    account's actual live quota can disagree - always trust the
    dashboard over any blog post or doc page.
  - gemma-4-26b-a4b-it / gemma-4-31b-it: confirmed via Aratrik's own
    dashboard to have real usable headroom, unlike every Gemini model
    tried so far. These are Gemma 4 (not Gemma 3 - Gemma 3 was
    superseded), served through the SAME Gemini API endpoint and SAME
    google-genai client - no separate integration needed. Source:
    https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

Gemma is now first in the fallback chain because it's the only model
confirmed to have real headroom on this account. The Gemini models
stay in the chain below Gemma as a fallback, but do not assume their
quotas without checking the dashboard again first - Google has changed
these numbers at least twice in the time this file has existed.

GEMINI_MODEL_FALLBACK_CHAIN is tried in order. If a model returns a
429 (quota exhausted) or 404 (model not found/discontinued), the next
model in the list is tried automatically, once per model, before giving
up. This means a discontinued model (like gemini-2.5-flash-lite became
in July 2026) doesn't silently break the app.
"""

import os
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv()

# Ordered by confirmed free-tier headroom, Gemma first since it's the
# only tier confirmed to have real room left on this account as of
# July 26, 2026. See module docstring before changing this order.
GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemma-4-26b-a4b-it",      # confirmed 10k+ RPD on this account - primary
    "gemma-4-31b-it",          # same family, same quota - fallback
]


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_bytes: bytes = None, image_mime_type: str = None) -> str:
        """
        Send prompt to the model, return plain text response.
        image_bytes/image_mime_type are optional - pass both together
        to include an image (e.g. a photographed diagram or handwritten
        question) alongside the text prompt. Providers that don't
        support images should raise NotImplementedError if an image is
        passed.
        """
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """
    Tries GEMINI_MODEL_FALLBACK_CHAIN in order (or just the one model
    if you pass model_name explicitly) and automatically falls back to
    the next model on a daily-quota 429, so one exhausted model doesn't
    stop generation mid-session.

    Gemma models (gemma-4-*) use this exact same client and generate_content
    call as Gemini models - same endpoint, same SDK, confirmed via
    https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api - so no
    separate provider class is needed for them.
    """

    def __init__(self, model_name: str = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")

        from google import genai

        self.client = genai.Client(api_key=api_key)
        # If a specific model was requested, only try that one (no
        # fallback) - this is what lets the CLI's "pick a model" menu
        # actually pin a choice instead of always overriding it.
        self.models_to_try = [model_name] if model_name else GEMINI_MODEL_FALLBACK_CHAIN
        self.last_model_used = None

    def generate(self, prompt: str, image_bytes: bytes = None, image_mime_type: str = None) -> str:
        from google.genai import types
        from google.genai.errors import ClientError

        contents = [prompt]
        if image_bytes is not None:
            if not image_mime_type:
                raise ValueError("image_mime_type is required when image_bytes is provided")
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))

        last_error = None
        for i, model_name in enumerate(self.models_to_try):
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )
                self.last_model_used = model_name
                return response.text
            except ClientError as e:
                last_error = e
                code = getattr(e, "code", None)
                # Continue on quota exhaustion (429) OR model-not-found (404) -
                # a discontinued model shouldn't break the whole chain.
                is_retryable = code in (429, 404) or "RESOURCE_EXHAUSTED" in str(e) or "NOT_FOUND" in str(e)
                if is_retryable and i < len(self.models_to_try) - 1:
                    continue  # try next model in the chain
                raise

        # Should be unreachable (loop either returns or raises), but
        # keep a safety net rather than silently returning None.
        if last_error:
            raise last_error
        raise RuntimeError("No Gemini/Gemma models available to try")


class NemotronProvider(LLMProvider):
    """
    Uses build.nvidia.com's OpenAI-compatible endpoint for Nemotron 3.
    Free tier: https://build.nvidia.com

    Does not support image input - Nemotron 3 Super via this endpoint
    is text-only. Raises NotImplementedError if an image is passed so
    the caller finds out immediately rather than the image silently
    being ignored.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, model_name: str = "nvidia/nemotron-3-super-120b-a12b"):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")

        self.api_key = api_key
        self.model_name = model_name
        self.last_model_used = model_name

    def generate(self, prompt: str, image_bytes: bytes = None, image_mime_type: str = None) -> str:
        if image_bytes is not None:
            raise NotImplementedError(
                "NemotronProvider does not support image input. Switch to the "
                "gemini provider for questions with an attached image."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_provider(name: str = None, model_name: str = None) -> LLMProvider:
    """
    Factory. Pass a name explicitly, or omit it to read PROVIDER from .env.
    model_name: for gemini, optionally pin one specific model instead of
    using the automatic fallback chain (e.g. if the user explicitly picks
    a specific model from a CLI menu, that choice should stick rather
    than silently falling back to the default chain).
    """
    name = (name or os.getenv("PROVIDER", "gemini")).lower()

    if name == "gemini":
        return GeminiProvider(model_name=model_name)
    elif name == "nemotron":
        return NemotronProvider(model_name=model_name) if model_name else NemotronProvider()
    else:
        raise ValueError(f"Unknown provider: {name}. Use 'gemini' or 'nemotron'.")
