# api/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AskRequest, AskResponse
from core import config
from services.embedding_client import EmbeddingClient
from services.index import FaissIndexStore
from services.index_builder import INDEX_PATH, CHUNK_PATH, main as build_index_main
from services.llm_client import OpenRouterClient
from services.rag_service import RAGService

app = FastAPI(title="GitHub Me Assistant", version="0.1.0")

# Allow your portfolio website to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire dependencies (simple manual DI)
embedder = EmbeddingClient(
    api_key=str(config.OPENROUTER_API_KEY),
    model=str(config.EMBED_MODEL),
    base_url="https://openrouter.ai/api/v1",
)
index_store = FaissIndexStore(index_path=INDEX_PATH, chunks_path=CHUNK_PATH, embedder=embedder)
llm = OpenRouterClient(api_key=str(config.OPENROUTER_API_KEY), model=str(config.OPENROUTER_MODEL))
rag = RAGService(index=index_store, llm=llm)

@app.get("/health")
def health():
    return {
        "ok": True,
        "index_ready": index_store.is_ready(),
        "indexed_chunks": len(index_store.chunks),
        "llm_model": config.OPENROUTER_MODEL,
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer, sources = rag.ask(
        req.question,
        top_k=req.top_k,
        conversation_id=req.conversation_id,
    )
    return AskResponse(
        answer=answer,
        sources=sources,
        conversation_id=req.conversation_id,
    )

@app.post("/reload-index")
def reload_index():
    index_store.reload()
    return {
        "ok": True,
        "index_ready": index_store.is_ready(),
        "indexed_chunks": len(index_store.chunks),
    }

@app.post("/build-index")
def build_index():
    build_index_main()
    index_store.reload()
    return {
        "ok": True,
        "index_ready": index_store.is_ready(),
        "indexed_chunks": len(index_store.chunks),
    }
