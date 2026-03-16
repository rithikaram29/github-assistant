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
        