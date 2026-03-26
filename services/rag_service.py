import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple
from services.index import IndexStore, Chunk
from services.llm_client import LLMClient
from api.schemas import Source

SYSTEM_RULES = """You are a polished assistant that answers questions about the user's GitHub work.
Write like a strong personal AI assistant, not like a retrieval system.
Answer directly and naturally.
Do not say phrases like "based on the context", "from the provided context", or "according to the snippets".
When listing projects or skills, lead with the answer itself.
Use short paragraphs by default. Use bullets only when they help readability.
If the indexed information is not enough to answer confidently, say that plainly in one sentence and suggest what to index next.
"""

MAX_HISTORY_TURNS = 6

@dataclass
class ConversationTurn:
    question: str
    answer: str

def format_history(history: List[ConversationTurn]) -> str:
    if not history:
        return "No prior conversation."

    lines: List[str] = []
    for idx, turn in enumerate(history, start=1):
        lines.append(f"User {idx}: {turn.question}")
        lines.append(f"Assistant {idx}: {turn.answer}")
    return "\n".join(lines)

def build_prompt(question: str, chunks: List[Chunk], history: List[ConversationTurn]):
    if not chunks:
        context = "No indexed GitHub context is available yet."
    else:
        formatted = []
        for i, ch in enumerate(chunks, start=1):
            title = ch.meta.get('title') or ch.meta.get("repo") or ch.id
            url = ch.meta.get("url") or ""
            formatted.append(f"[{i}] {title}\n{url}\n{ch.text}\n")
        context = "\n".join(formatted)
    return f"""{SYSTEM_RULES}

Conversation so far:
{format_history(history)}

Context:
{context}

Current user question: {question}

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
        self.conversations: Dict[str, Deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=MAX_HISTORY_TURNS)
        )
        
    def ask(self, question: str, top_k:int, conversation_id: str | None = None) -> Tuple[str, List[Source]]:
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

        history = list(self.conversations[conversation_id]) if conversation_id else []
        prompt = build_prompt(question, chunks, history)
        print(f"[RAG] sending prompt ({len(prompt)} chars) to LLM ...")
        answer = self.llm.generate(prompt)
        print(f"[RAG] answer ({len(answer)} chars): {answer[:120]!r}{'...' if len(answer) > 120 else ''}")

        if conversation_id:
            self.conversations[conversation_id].append(
                ConversationTurn(question=question, answer=answer)
            )

        return answer, chunks_to_sources(chunks)
        
