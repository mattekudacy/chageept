"""Seed crawler that discovers and indexes all pages from CHAGEE website.

Run as: `python -m scripts.seed_crawler` (from project root)
"""
import re
from urllib.parse import urljoin, urlparse

from dotenv import load_dotenv

load_dotenv()

from chageept.scraper import ScrapeTool
from chageept.retriever import SearchTool
import requests
from bs4 import BeautifulSoup


# Base URL and seed pages to start crawling from
BASE_URL = "https://global.chagee.com/ph/en"
SEED_URLS = [
    f"{BASE_URL}",
    f"{BASE_URL}/product",
    f"{BASE_URL}/about",
    f"{BASE_URL}/stores",
    f"{BASE_URL}/contact",
    f"{BASE_URL}/media-centre",
]

# Maximum pages to crawl (to avoid infinite loops)
MAX_PAGES = 50


def discover_links(url: str, base_domain: str) -> set:
    """Discover all internal links on a page."""
    links = set()
    try:
        response = requests.get(url, timeout=10, headers={
            "User-Agent": "CHAGEEPT-Bot/1.0 (Educational)"
        })
        soup = BeautifulSoup(response.text, "html.parser")
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()  # Remove whitespace
            
            # Skip empty hrefs
            if not href:
                continue
            
            # Convert relative URLs to absolute
            full_url = urljoin(url, href)
            parsed = urlparse(full_url)
            
            # Only keep links from same domain
            if parsed.netloc != base_domain:
                continue
            
            # Clean up the path - fix malformed /en/ph/en/ patterns
            path = parsed.path
            
            # Fix common URL issues
            path = re.sub(r'/en/ph/en/', '/ph/en/', path)  # Fix /en/ph/en/ -> /ph/en/
            path = re.sub(r'^/en(/ph/en/)', r'\1', path)   # Fix leading /en/ph/en/
            path = re.sub(r'/+', '/', path)                 # Fix double slashes
            
            # Only keep /ph/en/ pages
            if "/ph/en" not in path:
                continue
                
            # Skip non-page resources
            if path.endswith(('.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.svg', '.ico')):
                continue
            
            # Skip URLs with fragments or query params
            if "#" in full_url:
                continue
            
            # Normalize URL (remove trailing slash and whitespace)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/").strip()
            links.add(clean_url)
                
    except Exception as e:
        print(f"  ⚠️ Could not discover links from {url}: {e}")
    
    return links


def crawl_all_pages() -> set:
    """Discover all pages by crawling from seed URLs."""
    base_domain = urlparse(BASE_URL).netloc
    visited = set()
    to_visit = set(SEED_URLS)
    all_pages = set()
    
    print("🔍 Discovering pages...")
    
    while to_visit and len(all_pages) < MAX_PAGES:
        url = to_visit.pop()
        if url in visited:
            continue
            
        visited.add(url)
        all_pages.add(url)
        print(f"  Found: {url}")
        
        # Discover more links from this page
        new_links = discover_links(url, base_domain)
        to_visit.update(new_links - visited)
    
    print(f"\n📄 Discovered {len(all_pages)} pages total\n")
    return all_pages


def main():
    # Discover all pages
    all_urls = crawl_all_pages()
    
    # Initialize tools
    scraper = ScrapeTool()
    retriever = SearchTool(persist_directory="./chroma_db")
    
    total_docs = 0
    failed = []
    
    print("📥 Indexing content...")
    for url in sorted(all_urls):
        try:
            docs = scraper.scrape(url, throttle_seconds=1.0)
            if docs:
                retriever.add_documents(docs)
                total_docs += len(docs)
                print(f"  ✓ {len(docs)} chunks from {url}")
        except Exception as e:
            failed.append(url)
            print(f"  ✗ Failed: {url} - {e}")

    print(f"\n{'='*50}")
    print(f"🎉 Seed crawl complete!")
    print(f"   Pages crawled: {len(all_urls)}")
    print(f"   Documents indexed: {total_docs}")
    print(f"   Failed: {len(failed)}")
    print(f"   Vector DB: ./chroma_db")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
