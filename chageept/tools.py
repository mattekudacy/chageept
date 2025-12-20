from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Document:
    id: str
    title: str
    url: str
    category: Optional[str]
    text: str
    metadata: Dict


@dataclass
class SearchResult:
    document: Document
    score: float
    snippet: str


class ScrapeNotAllowed(Exception):
    pass
