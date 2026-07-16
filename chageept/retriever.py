import os
import time
from typing import List, Optional
import chromadb
from openai import OpenAI, RateLimitError
from .tools import SearchResult, Document

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_BASE_DELAY = 5


class SearchTool:
    """SearchTool powered by ChromaDB and Gemini's hosted embedding API.

    Methods:
      - search(query, top_k) -> List[SearchResult]
      - add_documents(docs)
    """

    def __init__(self, collection_name: str = "chageept_docs", persist_directory: str = "./chroma_db"):
        self.embedding_client = OpenAI(
            base_url=GEMINI_BASE_URL, api_key=os.getenv("GEMINI_API_KEY")
        )
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

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts via Gemini's embedding API, batching to stay
        within request limits and retrying with backoff on rate limits."""
        embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            for attempt in range(EMBEDDING_MAX_RETRIES):
                try:
                    response = self.embedding_client.embeddings.create(
                        model=EMBEDDING_MODEL, input=batch
                    )
                    embeddings.extend(item.embedding for item in response.data)
                    break
                except RateLimitError:
                    if attempt == EMBEDDING_MAX_RETRIES - 1:
                        raise
                    time.sleep(EMBEDDING_RETRY_BASE_DELAY * (2 ** attempt))
        return embeddings

    def add_documents(self, docs: List[Document]):
        """Embed and store documents in ChromaDB."""
        if not docs:
            return
        ids = [d.id for d in docs]
        texts = [d.text for d in docs]
        embeddings = self._embed(texts)
        metadatas = [
            {"title": d.title, "url": d.url, "category": d.category or ""}
            for d in docs
        ]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        for d in docs:
            self._doc_registry[d.id] = d

    def search(self, query: str, top_k: int = 4, category: Optional[str] = None) -> List[SearchResult]:
        """Semantic search using embeddings, optionally restricted to a category."""
        query_embedding = self._embed([query])[0]
        query_kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if category:
            query_kwargs["where"] = {"category": category}
        results_raw = self.collection.query(**query_kwargs)

        search_results = []
        if results_raw["ids"] and results_raw["ids"][0]:
            for i, doc_id in enumerate(results_raw["ids"][0]):
                # ChromaDB cosine distance = 1 - cosine_similarity, so similarity = 1 - distance
                distance = results_raw["distances"][0][i] if "distances" in results_raw else 1.0
                score = 1.0 - distance
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                
                doc = self._doc_registry.get(doc_id)
                if doc:
                    snippet = doc.text[:300].replace("\n", " ")
                    search_results.append(SearchResult(document=doc, score=score, snippet=snippet))
        
        return search_results
