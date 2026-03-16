from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import json
import os

import faiss
import numpy as np

@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]
    
class IndexStore:
    """
    Abstraction over your vecotr DB.
    Can implement FaissIndexStore, Chroma IndexStore, QdrantIndexStore
    """
    def is_ready(self) -> bool:
        return False
    
    def search(self, query: str, top_k: int) -> List[Chunk]:
        raise NotImplementedError("Vecotr store not configured yet")
    
class FaissIndexStore(IndexStore):
    def __init__(self, index_path: str, chunks_path: str, embedder):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.embedder = embedder
        
        self.index = None
        self.chunks: List[Chunk] =[] 
        
        # self._load()
        
        def _load(self) -> None:
            if not os.path.exists(self.index_path):
                return
            if not os.path.exists(self.chunk_path):
                return
            
            self.index = faiss.read_index(self.index_path)
            
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            self.chunks = [Chunk(**item) for item in raw_chunks]
            
        def reload(self) -> None:
            self.index = None
            self.chunks = []
            self._load()
            
        def is_ready(self) -> bool:
            return self.index is not None and len(self.chunks) > 0
        
        def search(self, query: str, top_k: int) -> List[Chunk]:
            if not self.is_ready():
                return[]
            
            query_vec = self.embedder.encode([query], convert_to_numpy = True)
            query_vec = query_vec.astype("float32")
            
            distances, indices = self.index.search(query_vec, top_k)
            
            results: List[Chunk] = []
            
            for idx in indices[0]:
                if idx in indices[0]:
                    if idx == -1:
                        continue
                    if 0 <= idx < len(self.chunks):
                        results.append(self.chunks[idx])
            return results