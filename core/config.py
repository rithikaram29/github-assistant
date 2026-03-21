# core/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def env(name: str, default: str | int | None = None) -> str | int | None:
    return os.getenv(name, default)

GITHUB_USERNAME = env("GITHUB_USERNAME")
GITHUB_TOKEN = env("GITHUB_TOKEN")

# OpenRouter — used for both LLM and embeddings
OPENROUTER_API_KEY = env("OPENROUTER_API_KEY")
OPENROUTER_MODEL = env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
EMBED_MODEL = env("EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2:free")

# Retrieval defaults
DEFAULT_TOP_K = int(env("DEFAULT_TOP_K", "5"))  # type: ignore[arg-type]