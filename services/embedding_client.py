from __future__ import annotations

import json
import urllib.request
from typing import List

import numpy as np


class EmbeddingClient:
    """
    OpenAI-compatible embedding API client (POST /embeddings).
    Works with OpenAI, OpenRouter, Jina, or any compatible provider.
    """

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return np.array(all_embeddings, dtype="float32")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        # nvidia/llama-nemotron-embed-vl-* requires multimodal content format.
        # Other models accept plain strings. Wrap automatically based on model name.
        if "nemotron" in self.model:
            input_data = [{"content": [{"type": "text", "text": t}]} for t in texts]
        else:
            input_data = texts
        payload = {"input": input_data, "model": self.model, "encoding_format": "float"}
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


class OllamaEmbeddingClient:
    """
    Ollama embedding client — calls POST /api/embed.

    Works with both:
    - Hosted Ollama (https://ollama.com) — pass api_key for Bearer auth
    - Local Ollama (http://localhost:11434) — api_key is ignored
    """

    def __init__(self, model: str, base_url: str = "https://ollama.com", api_key: str = ""):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def encode(
        self,
        texts: List[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 64,
    ) -> np.ndarray:
        all_embeddings: List[List[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return np.array(all_embeddings, dtype="float32")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": texts}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"https://ollama.com/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=240) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Ollama returns {"embeddings": [[...], [...]]}
        return data["embeddings"]
