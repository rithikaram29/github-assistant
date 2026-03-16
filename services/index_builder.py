from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any

import faiss
import numpy as np
import requests

from sentence_transformers import SentenceTransformer

from core.config import GITHUB_USERNAME, GITHUB_TOKEN, EMBED_MODEL

DATA_DIR = "data"
CHUNK_PATH = os.path.join(DATA_DIR,"chunks.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

def github_headers() -> Dict[str,str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-me-assistant"
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

def fetch_repos(username: str) -> List[Dict[str, Any]]:
    url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    resp = requests.get(url, headers=github_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_readme(owner: str, repo: str) -> str:
    