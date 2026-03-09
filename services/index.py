from dataclasses import dataclass
from typing import List, Optional, Dict, Any

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