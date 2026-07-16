import re
from dataclasses import dataclass
from typing import Dict, List, Optional

LIST_KEYWORDS = [
    "list", "all", "menu", "items", "drinks", "what do you have",
    "what are", "show me", "options", "available",
]
LIST_QUERY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in LIST_KEYWORDS) + r")\b", re.IGNORECASE
)


def is_list_query(query: str) -> bool:
    """Whether a query is asking for a list/enumeration (menu, drinks,
    stores, etc.) rather than a single fact - used to widen retrieval and
    raise the answer's token budget."""
    return bool(LIST_QUERY_PATTERN.search(query))


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
