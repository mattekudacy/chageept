"""Tavily web search tool - a fallback used only when the CHAGEE knowledge
base doesn't have a good answer.
"""
import os
from typing import Dict, List, Optional
import requests

TAVILY_API_URL = "https://api.tavily.com/search"


class TavilySearchTool:
    """Thin wrapper around the Tavily Search API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """Run a web search and return a list of {title, url, content}."""
        if not self.api_key:
            return []

        response = requests.post(
            TAVILY_API_URL,
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        return [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
            }
            for item in data.get("results", [])
        ]
