"""
llm_providers.py

Provider-agnostic interface for the generation step of the RAG pipeline.
Swap backends by changing PROVIDER in your .env, no other code changes.

Requires a .env file in the repo root with:
  GEMINI_API_KEY=your_key_here
  NVIDIA_API_KEY=your_key_here      # from build.nvidia.com, for Nemotron 3
  PROVIDER=gemini                   # or: nemotron
"""

import os
from abc import ABC, abstractmethod

import requests
from dotenv import load_dotenv

load_dotenv()


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Send prompt to the model, return plain text response."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-3.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set in .env")

        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text


class NemotronProvider(LLMProvider):
    """
    Uses build.nvidia.com's OpenAI-compatible endpoint for Nemotron 3.
    Free tier: https://build.nvidia.com
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, model_name: str = "nvidia/nemotron-3-super"):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY not set in .env")

        self.api_key = api_key
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        resp = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def get_provider(name: str = None) -> LLMProvider:
    """
    Factory. Pass a name explicitly, or omit it to read PROVIDER from .env.
    """
    name = (name or os.getenv("PROVIDER", "gemini")).lower()

    if name == "gemini":
        return GeminiProvider()
    elif name == "nemotron":
        return NemotronProvider()
    else:
        raise ValueError(f"Unknown provider: {name}. Use 'gemini' or 'nemotron'.")
