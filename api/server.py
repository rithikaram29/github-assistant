# api/server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import AskRequest, AskResponse
from core import config
from services.index import IndexStore
from services.llm_client import OllamaClient
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
index_store = IndexStore()  # stub for now
llm = OllamaClient(base_url=str(config.OLLAMA_BASE_URL) , model=str(config.OLLAMA_MODEL))
rag = RAGService(index=index_store, llm=llm)

@app.get("/health")
def health():
    return {
        "ok": True,
        "index_ready": index_store.is_ready(),
        "ollama_model": config.OLLAMA_MODEL,
    }

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    answer, sources = rag.ask(req.question, top_k=req.top_k)
    return AskResponse(answer=answer, sources=sources)