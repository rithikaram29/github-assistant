# core/config.py
import os

def env(name: str, default: str | int | None = None) -> str | int | None:
    return os.getenv(name, default)

GITHUB_USERNAME = env("GITHUB_USERNAME")
GITHUB_TOKEN = env("GITHUB_TOKEN")

# LLM (local for now)
OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = env("OLLAMA_MODEL", "qwen2.5:3b")

# Retrieval defaults
DEFAULT_TOP_K = (env("DEFAULT_TOP_K", "5"))