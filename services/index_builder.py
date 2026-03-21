from __future__ import annotations

import json
import os
import re
from typing import List, Dict, Any

import faiss
import numpy as np
import requests

from core.config import GITHUB_USERNAME, GITHUB_TOKEN, EMBED_MODEL, OPENROUTER_API_KEY
from services.embedding_client import EmbeddingClient

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
    # Use authenticated endpoint when a token is available — returns private repos too
    if GITHUB_TOKEN:
        url = "https://api.github.com/user/repos?per_page=100&sort=updated&affiliation=owner"
    else:
        url = f"https://api.github.com/users/{username}/repos?per_page=100&sort=updated"
    resp = requests.get(url, headers=github_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()

def fetch_readme(owner: str, repo: str) -> str:
    url = f"https://api.github.com/repos{owner}/{repo}/readme"
    resp = requests.get(url, headers=github_headers(), timeout=30)
    
    if resp.status_code == 404:
        return ""
    
    resp.raise_for_status()
    data = resp.json()
    
    download_url = data.get("download_url")
    if not download_url:
        return ""
    
    raw_rep = requests.get(download_url, timeout = 30)
    raw_rep.raise_for_status()
    return raw_rep.text


def fetch_languages(owner: str, repo: str) -> List[str]:
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    resp = requests.get(url,headers=github_headers(), timeout=30)
    
    if resp.status_code != 200:
        return []
    
    data = resp.json()
    
    return list(data.keys())

def guess_stack(description: str, readme: str, languages:List[str] | Any, topics: List[str]) -> List[str]:
    text=" ".join([description or "", readme or "", " ".join(languages)," ".join(topics)]).lower()
    
    candidates = [
        "python",
        "fastapi",
        "flask",
        "django",
        "javascript",
        "typescript",
        "react",
        "next.js",
        "node.js",
        "express",
        "java",
        "spring",
        "kotlin",
        "swift",
        "flutter",
        "dart",
        "supabase",
        "firebase",
        "postgresql",
        "mysql",
        "mongodb",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "iot",
        "machine learning",
        "llm",
        "typerscript",
        "javascript"
    ]
    
    found = [item for item in candidates if item in text]
    
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower not in found:
            found.append(lang_lower)
            
    return found[:15]

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n","\n",text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_repo_document(repo:Dict[str, Any], readme: str, languages: List[str]) -> Dict[str, Any]:
    owner = repo["owner"]["login"]
    repo_name = repo["name"]
    description = repo.get("description") or ""
    topics = repo.get("topics") or ""
    html_url = repo.get("html_url")
    updated_at = repo.get("updated_at")
    
    stack = guess_stack(description, readme, languages, topics)
    
    text= f"""
Repository: {repo_name}
Owner: {owner}
Description: {description}
Topics: {", ".join(topics)}
Languages: {", ".join(languages)}
Tech stack: {", ".join(stack)}
URL: {html_url}

README:
{readme}
""".strip()

    return {
        "id": f"{owner}/{repo_name}",
        "text": clean_text(text),
        "meta": {
            "title": repo_name,
            "repo": f"{owner}/{repo_name}",
            "url": html_url,
            "topics": topics,
            "languages": languages,
            "tech_stack": stack,
            "updated_at": updated_at,
        },
    }

def save_index(chunks: List[Dict[str, Any]], embedder) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    texts = [c["text"] for c in chunks]

    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    # cosine similarity via normalized inner product
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(CHUNK_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        

def main() -> None:
    if not GITHUB_USERNAME:
        raise ValueError("GITHUB_USERNAME is not set")

    repos = fetch_repos(str(GITHUB_USERNAME))
    chunks: List[Dict[str, Any]] = []

    for repo in repos:
        owner = repo["owner"]["login"]
        repo_name = repo["name"]

        try:
            readme = fetch_readme(owner, repo_name)
        except Exception:
            readme = ""

        try:
            languages = fetch_languages(owner, repo_name)
        except Exception:
            languages = []

        chunk = build_repo_document(repo, readme, languages)
        chunks.append(chunk)

    embedder = EmbeddingClient(
        api_key=str(OPENROUTER_API_KEY),
        model=str(EMBED_MODEL),
        base_url="https://openrouter.ai/api/v1",
    )
    save_index(chunks, embedder)
    print(f"Built index with {len(chunks)} repositories")
    
    

if __name__ == "__main__":
    main()