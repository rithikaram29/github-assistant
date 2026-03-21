from typing import Optional
import json
import urllib.request

class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

class OllamaClient(LLMClient):          ## use free api tokens
    def __init__(self, base_url: str,model: str, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        req = urllib.request.Request(
            url,
            data= json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Ollama returns {"response": "...", ...}
        return (data.get("response") or "").strip()


class OpenRouterClient(LLMClient):
    """
    LLM client backed by OpenRouter (https://openrouter.ai).
    Uses the OpenAI-compatible /chat/completions endpoint.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key: str, model: str, timeout_s: int = 120):
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        req = urllib.request.Request(
            self.BASE_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            return ""
        