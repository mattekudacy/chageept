from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from .tools import SearchResult, Document


class SearchTool:
    """SearchTool powered by ChromaDB and sentence-transformers embeddings.

    Methods:
      - search(query, top_k) -> List[SearchResult]
      - add_documents(docs)
    """

    def __init__(self, collection_name: str = "chageept_docs", persist_directory: str = "./chroma_db"):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Use cosine similarity for semantic search
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._doc_registry = {}  # map doc_id -> Document
        
        # Rebuild doc_registry from existing ChromaDB data
        self._rebuild_registry()
    
    def _rebuild_registry(self):
        """Rebuild the document registry from ChromaDB metadata."""
        try:
            results = self.collection.get()
            if results and results["ids"]:
                from .tools import Document
                for i, doc_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    text = results["documents"][i] if results["documents"] else ""
                    
                    doc = Document(
                        id=doc_id,
                        title=metadata.get("title", ""),
                        url=metadata.get("url", ""),
                        category=metadata.get("category"),
                        text=text,
                        metadata=metadata
                    )
                    self._doc_registry[doc_id] = doc
        except Exception:
            pass  # Empty collection or error, start fresh

    def add_documents(self, docs: List[Document]):
        """Embed and store documents in ChromaDB."""
        if not docs:
            return
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        metadatas = [
            {"title": d.title, "url": d.url, "category": d.category or ""}
            for d in docs
        ]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        for d in docs:
            self._doc_registry[d.id] = d

    def search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """Semantic search using embeddings."""
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()[0]
        results_raw = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        search_results = []
        if results_raw["ids"] and results_raw["ids"][0]:
            for i, doc_id in enumerate(results_raw["ids"][0]):
                # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
                # Convert to similarity score (1 = identical, 0 = orthogonal, -1 = opposite)
                distance = results_raw["distances"][0][i] if "distances" in results_raw else 1.0
                score = 1.0 - (distance / 2.0)  # Normalize to 0-1 range
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                
                doc = self._doc_registry.get(doc_id)
                if doc:
                    snippet = doc.text[:300].replace("\n", " ")
                    search_results.append(SearchResult(document=doc, score=score, snippet=snippet))
        
        return search_results
