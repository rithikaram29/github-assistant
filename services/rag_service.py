import traceback
from typing import List, Tuple
from services.index import IndexStore, Chunk
from services.llm_client import LLMClient
from api.schemas import Source

SYSTEM_RULES = """You are an assistant that answers questions about the user's GitHub work.
Use the provided context snippets when available.
If the answer is not supported by the context, say you don't know and suggest what to index.
Keep answers concise.
"""

def build_prompt(question: str, chunks: List[Chunk]):
    if not chunks:
        context = "No indexed Github context is available yet."
    else:
        formatted = []
        for i, ch in enumerate(chunks, start=1):
            title = ch.meta.get('title') or ch.meta.get("repo") or ch.id
            url = ch.meta.get("url") or ""
            formatted.append(f"[{i}] {title}\n{url}\n{ch.text}\n")
        context = "\n".join(formatted)
    return f"""{SYSTEM_RULES}

Context:
{context}

Question: {question}

Answer:"""

def chunks_to_sources(chunks: List[Chunk]) -> List[Source]:
    sources: List[Source] = []
    for ch in chunks:
        sources.append(
            Source(
                title=str(ch.meta.get("title") or ch.meta.get("repo") or ch.id),
                url=ch.meta.get("url"),
                repo=ch.meta.get("repo"),
                path=ch.meta.get("path"),
                extra={k: v for k, v in ch.meta.items() if k not in {"title", "url", "repo", "path"}},
            )
        )
    return sources

class RAGService:
    def __init__(self, index: IndexStore, llm: LLMClient):
        self.index = index
        self.llm = llm
        
    def ask(self, question: str, top_k:int) -> Tuple[str, List[Source]]:
        print(f"\n[RAG] question: {question!r}  top_k={top_k}")

        chunks: List[Chunk] = []
        try:
            if self.index.is_ready():
                chunks = self.index.search(question, top_k=top_k)
                print(f"[RAG] retrieved {len(chunks)} chunk(s):")
                for i, ch in enumerate(chunks, 1):
                    title = ch.meta.get("title") or ch.meta.get("repo") or ch.id
                    print(f"  [{i}] {title}")
            else:
                print("[RAG] index not ready — skipping retrieval")
        except Exception as e:
            print(f"[RAG] retrieval error: {type(e).__name__}: {e}")
            traceback.print_exc()
            chunks = []

        prompt = build_prompt(question, chunks)
        print(f"[RAG] sending prompt ({len(prompt)} chars) to LLM ...")
        answer = self.llm.generate(prompt)
        print(f"[RAG] answer ({len(answer)} chars): {answer[:120]!r}{'...' if len(answer) > 120 else ''}")
        return answer, chunks_to_sources(chunks)
        