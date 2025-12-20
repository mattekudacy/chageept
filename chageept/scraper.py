import time
import uuid
from typing import List
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from urllib import robotparser

from .tools import Document, ScrapeNotAllowed


class ScrapeTool:
    """Simple ScrapeTool that respects robots.txt and extracts page text.

    Use methods:
      - is_allowed(url)
      - scrape(url) -> List[Document]
    """

    def __init__(self, user_agent: str = "CHAGEEPT-bot/0.1 (+https://chagee.com)"):
        self.user_agent = user_agent
        self._rp_cache = {}

    def _get_robot_parser(self, base_url: str) -> robotparser.RobotFileParser:
        if base_url in self._rp_cache:
            return self._rp_cache[base_url]
        rp = robotparser.RobotFileParser()
        robots_url = base_url.rstrip("/") + "/robots.txt"
        try:
            rp.set_url(robots_url)
            rp.read()
        except Exception:
            rp = robotparser.RobotFileParser()
            rp.parse(())
        self._rp_cache[base_url] = rp
        return rp

    def is_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        rp = self._get_robot_parser(base)
        return rp.can_fetch(self.user_agent, url)

    def fetch_page(self, url: str, timeout: int = 10) -> str:
        headers = {"User-Agent": self.user_agent}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.text

    def extract_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        parts = []
        for tag in soup.select("h1,h2,h3,p,li"):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                parts.append(text)
        return "\n".join(parts)

    def chunk_text(self, text: str, max_chars: int = 2000) -> List[str]:
        # Simple chunker: split by paragraph and accumulate
        paras = [p.strip() for p in text.split("\n") if p.strip()]
        chunks = []
        cur = []
        cur_len = 0
        for p in paras:
            if cur_len + len(p) + 1 > max_chars and cur:
                chunks.append("\n".join(cur))
                cur = [p]
                cur_len = len(p)
            else:
                cur.append(p)
                cur_len += len(p) + 1
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def scrape(self, url: str, throttle_seconds: float = 1.0) -> List[Document]:
        if not self.is_allowed(url):
            raise ScrapeNotAllowed(f"Scraping disallowed by robots.txt: {url}")
        time.sleep(throttle_seconds)
        html = self.fetch_page(url)
        text = self.extract_text(html)
        chunks = self.chunk_text(text)
        docs = []
        for i, c in enumerate(chunks):
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    title=(url.split("/")[-1] or url),
                    url=url,
                    category=None,
                    text=c,
                    metadata={"source_type": "on_demand"},
                )
            )
        return docs
