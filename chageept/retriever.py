import os
import time
from typing import List, Optional
from openai import OpenAI, RateLimitError
from qdrant_client import QdrantClient, models
from .tools import SearchResult, Document

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 3072
EMBEDDING_BATCH_SIZE = 50
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_BASE_DELAY = 5


class SearchTool:
    """SearchTool powered by Qdrant and Gemini's hosted embedding API.

    Methods:
      - search(query, top_k) -> List[SearchResult]
      - add_documents(docs)
    """

    def __init__(self, collection_name: str = "chageept_docs"):
        self.embedding_client = OpenAI(
            base_url=GEMINI_BASE_URL, api_key=os.getenv("GEMINI_API_KEY")
        )
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY")
        )
        self.collection_name = collection_name
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIMENSIONS, distance=models.Distance.COSINE
                ),
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="category",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

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
        """Embed and store documents in Qdrant."""
        if not docs:
            return
        texts = [d.text for d in docs]
        embeddings = self._embed(texts)
        points = [
            models.PointStruct(
                id=d.id,
                vector=embedding,
                payload={
                    "title": d.title,
                    "url": d.url,
                    "category": d.category or "",
                    "text": d.text,
                },
            )
            for d, embedding in zip(docs, embeddings)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 4, category: Optional[str] = None) -> List[SearchResult]:
        """Semantic search using embeddings, optionally restricted to a category."""
        query_embedding = self._embed([query])[0]
        query_filter = None
        if category:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
            )
        results_raw = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        search_results = []
        for point in results_raw.points:
            payload = point.payload or {}
            text = payload.get("text", "")
            doc = Document(
                id=str(point.id),
                title=payload.get("title", ""),
                url=payload.get("url", ""),
                category=payload.get("category"),
                text=text,
                metadata=payload,
            )
            score = max(0.0, min(1.0, point.score))  # Clamp to [0, 1]
            snippet = text[:300].replace("\n", " ")
            search_results.append(SearchResult(document=doc, score=score, snippet=snippet))

        return search_results

    def count(self) -> int:
        """Number of documents currently stored."""
        return self.client.count(collection_name=self.collection_name).count
