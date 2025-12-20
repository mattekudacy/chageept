import time
import uuid
import json
import re
from typing import List, Dict, Optional
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup, Tag
from urllib import robotparser

from .tools import Document, ScrapeNotAllowed


class ScrapeTool:
    """Advanced ScrapeTool with comprehensive content extraction.

    Features:
    - Respects robots.txt
    - Extracts structured data (JSON-LD, meta tags)
    - Identifies main content areas
    - Extracts product information
    - Captures image descriptions
    - Cleans and chunks text intelligently
    """

    def __init__(self, user_agent: str = "CHAGEEPT-bot/1.0 (+https://chagee.com)"):
        self.user_agent = user_agent
        self._rp_cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        })

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

    def fetch_page(self, url: str, timeout: int = 15) -> str:
        r = self.session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.text

    def extract_meta_data(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from page head."""
        meta = {
            "title": "",
            "description": "",
            "keywords": [],
            "og_title": "",
            "og_description": "",
        }
        
        # Title
        title_tag = soup.find("title")
        if title_tag:
            meta["title"] = title_tag.get_text(strip=True)
        
        # Meta tags
        for tag in soup.find_all("meta"):
            name = tag.get("name", "").lower()
            prop = tag.get("property", "").lower()
            content = tag.get("content", "")
            
            if name == "description" or prop == "og:description":
                meta["description"] = content
            elif name == "keywords":
                meta["keywords"] = [k.strip() for k in content.split(",")]
            elif prop == "og:title":
                meta["og_title"] = content
        
        return meta

    def extract_json_ld(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract JSON-LD structured data."""
        json_ld_data = []
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, list):
                    json_ld_data.extend(data)
                else:
                    json_ld_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        return json_ld_data

    def extract_product_info(self, soup: BeautifulSoup, url: str) -> List[Dict]:
        """Extract product/menu item information."""
        products = []
        
        # Look for product cards/items with various class patterns
        product_selectors = [
            "[class*='product']",
            "[class*='menu-item']",
            "[class*='card']",
            "[class*='item']",
            "[data-product]",
            "article",
        ]
        
        for selector in product_selectors:
            for item in soup.select(selector):
                # Skip if it's a navigation or header element
                if item.find_parent(["nav", "header", "footer"]):
                    continue
                
                # Extract product name
                name_tag = item.select_one("h1, h2, h3, h4, [class*='name'], [class*='title']")
                name = name_tag.get_text(strip=True) if name_tag else ""
                
                # Extract description
                desc_tag = item.select_one("p, [class*='desc'], [class*='description']")
                desc = desc_tag.get_text(strip=True) if desc_tag else ""
                
                # Extract price if available
                price_tag = item.select_one("[class*='price']")
                price = price_tag.get_text(strip=True) if price_tag else ""
                
                # Extract image alt text
                img_tag = item.select_one("img")
                img_alt = img_tag.get("alt", "") if img_tag else ""
                
                if name and len(name) > 3:
                    products.append({
                        "name": name,
                        "description": desc or img_alt,
                        "price": price,
                    })
        
        # Deduplicate by name
        seen = set()
        unique_products = []
        for p in products:
            if p["name"] not in seen:
                seen.add(p["name"])
                unique_products.append(p)
        
        return unique_products

    def extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content with intelligent filtering."""
        # Remove unwanted elements
        for tag in soup.select("script, style, nav, footer, header, noscript, iframe, svg, [hidden]"):
            tag.decompose()
        
        # Remove elements that are likely menus/navigation
        for tag in soup.select("[class*='nav'], [class*='menu']:not([class*='menu-item']), [class*='sidebar'], [class*='footer'], [class*='header']"):
            if tag.name not in ["main", "article", "section"]:
                tag.decompose()
        
        # Try to find main content area
        main_selectors = [
            "main",
            "article", 
            "[role='main']",
            ".content",
            "#content",
            ".main-content",
            "#main",
            ".page-content",
        ]
        
        content_area = None
        for selector in main_selectors:
            content_area = soup.select_one(selector)
            if content_area:
                break
        
        if not content_area:
            content_area = soup.body if soup.body else soup
        
        return content_area

    def extract_text_blocks(self, element: Tag) -> List[str]:
        """Extract meaningful text blocks from an element."""
        blocks = []
        
        # Get all text-containing elements
        for tag in element.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "span", "div", "td", "th"]):
            # Skip if parent is already processed
            if tag.find_parent(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
                continue
            
            text = tag.get_text(separator=" ", strip=True)
            
            # Filter out short or meaningless text
            if len(text) < 10:
                continue
            if text.lower() in ["menu", "home", "contact", "about", "search", "login", "sign up"]:
                continue
            
            # Check if this is actual content (not just a single word repeated)
            words = text.split()
            if len(words) < 3:
                continue
            
            blocks.append(text)
        
        return blocks

    def extract_images_with_context(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image descriptions and alt text."""
        image_info = []
        
        for img in soup.find_all("img"):
            alt = img.get("alt", "").strip()
            title = img.get("title", "").strip()
            
            # Get surrounding context
            parent = img.find_parent(["figure", "div", "article", "section"])
            caption = ""
            if parent:
                figcaption = parent.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text(strip=True)
            
            # Combine available info
            info = " - ".join(filter(None, [alt, title, caption]))
            if info and len(info) > 5:
                image_info.append(f"[Image: {info}]")
        
        return image_info

    def clean_and_dedupe(self, texts: List[str]) -> List[str]:
        """Clean and deduplicate text list."""
        cleaned = []
        seen = set()
        
        for text in texts:
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Skip duplicates (using first 50 chars as key)
            key = text[:50].lower()
            if key in seen:
                continue
            seen.add(key)
            
            # Skip if too short after cleaning
            if len(text) < 15:
                continue
            
            cleaned.append(text)
        
        return cleaned

    def format_content(self, meta: Dict, products: List[Dict], text_blocks: List[str], images: List[str]) -> str:
        """Format all extracted content into a coherent document."""
        sections = []
        
        # Add title and description
        title = meta.get("og_title") or meta.get("title", "")
        if title:
            # Clean up title (remove site name)
            title = title.split("|")[0].split("-")[0].strip()
            sections.append(f"# {title}")
        
        desc = meta.get("description", "")
        if desc:
            sections.append(f"\n{desc}\n")
        
        # Add product information
        if products:
            sections.append("\n## Products/Menu Items:\n")
            for p in products:
                product_text = f"**{p['name']}**"
                if p['description']:
                    product_text += f": {p['description']}"
                if p['price']:
                    product_text += f" - {p['price']}"
                sections.append(product_text)
        
        # Add main content
        if text_blocks:
            sections.append("\n## Content:\n")
            sections.extend(text_blocks)
        
        # Add image descriptions
        if images:
            sections.append("\n## Visual Information:\n")
            sections.extend(images[:10])  # Limit to 10 images
        
        return "\n".join(sections)

    def chunk_text(self, text: str, max_chars: int = 2500, overlap: int = 300) -> List[str]:
        """Smart chunking with overlap for better context."""
        if len(text) <= max_chars:
            return [text]
        
        # Split by paragraphs/sections first
        paragraphs = re.split(r'\n\n+', text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            # If single paragraph exceeds max, split by sentences
            if para_len > max_chars:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    if current_len + len(sent) > max_chars and current_chunk:
                        chunks.append("\n".join(current_chunk))
                        # Keep last part for overlap
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text] if len(overlap_text) < overlap else []
                        current_len = len(overlap_text) if current_chunk else 0
                    current_chunk.append(sent)
                    current_len += len(sent) + 1
            elif current_len + para_len > max_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                # Keep last paragraph for overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if len(overlap_text) < overlap else [para]
                current_len = len(overlap_text) + para_len if len(overlap_text) < overlap else para_len
            else:
                current_chunk.append(para)
                current_len += para_len + 1
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks

    def determine_category(self, url: str, meta: Dict) -> str:
        """Determine page category from URL and metadata."""
        url_lower = url.lower()
        title_lower = meta.get("title", "").lower()
        
        if any(x in url_lower for x in ["product", "menu", "tea", "milk", "frappe", "latte"]):
            return "menu"
        elif "store" in url_lower or "location" in url_lower:
            return "stores"
        elif "about" in url_lower or "history" in url_lower:
            return "about"
        elif "contact" in url_lower:
            return "contact"
        elif any(x in url_lower for x in ["media", "news", "blog", "article"]):
            return "news"
        elif "sustainability" in url_lower:
            return "sustainability"
        elif "privacy" in url_lower or "legal" in url_lower or "terms" in url_lower:
            return "legal"
        else:
            return "general"

    def generate_descriptive_title(self, url: str, meta: Dict, category: str) -> str:
        """Generate a clear, descriptive title based on URL and content."""
        # Parse URL path for context
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p and p not in ["ph", "en"]]
        
        # Map URL patterns to descriptive titles
        title_mappings = {
            # Product pages
            "fresh-milk-tea-series": "Fresh Milk Tea Series Menu",
            "brewed-tea-series": "Brewed Tea Series Menu",
            "snowy-frappe-series": "Snowy Frappé Series Menu",
            "teaspresso-latte": "Teaspresso Latte Menu",
            "teaspresso-frappe": "Teaspresso Frappé Menu",
            "product": "CHAGEE Menu & Products",
            # About pages
            "about": "About CHAGEE",
            "history": "CHAGEE History",
            # Other pages
            "stores": "CHAGEE Store Locations",
            "contact": "Contact CHAGEE Philippines",
            "media-centre": "CHAGEE News & Media",
            "privacy-policy": "CHAGEE Privacy Policy",
            "terms-of-use": "CHAGEE Terms of Use",
            "privacy": "CHAGEE Legal Information",
        }
        
        # Check path parts for matching titles
        for part in reversed(path_parts):  # Check from most specific to least
            if part in title_mappings:
                return title_mappings[part]
            # Handle year pages (history)
            if part.isdigit() and len(part) == 4:
                return f"CHAGEE History - {part}"
        
        # Fallback to category-based title
        category_titles = {
            "menu": "CHAGEE Menu",
            "stores": "CHAGEE Stores",
            "about": "About CHAGEE",
            "contact": "Contact CHAGEE",
            "news": "CHAGEE News",
            "legal": "CHAGEE Legal",
            "general": "CHAGEE Philippines",
        }
        
        return category_titles.get(category, "CHAGEE Philippines")

    def scrape(self, url: str, throttle_seconds: float = 1.0) -> List[Document]:
        """Scrape a URL and return structured documents."""
        if not self.is_allowed(url):
            raise ScrapeNotAllowed(f"Scraping disallowed by robots.txt: {url}")
        
        time.sleep(throttle_seconds)
        html = self.fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract all components
        meta = self.extract_meta_data(soup)
        json_ld = self.extract_json_ld(soup)
        products = self.extract_product_info(soup, url)
        
        # Get main content
        content_area = self.extract_main_content(soup)
        text_blocks = self.extract_text_blocks(content_area)
        text_blocks = self.clean_and_dedupe(text_blocks)
        
        # Get image info
        images = self.extract_images_with_context(soup, url)
        
        # Combine everything
        full_content = self.format_content(meta, products, text_blocks, images)
        
        # Chunk the content
        chunks = self.chunk_text(full_content)
        
        # Determine category and generate descriptive title
        category = self.determine_category(url, meta)
        base_title = self.generate_descriptive_title(url, meta, category)
        
        # Create documents
        docs = []
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # Skip very short chunks
                continue
            
            # Only add part number if multiple chunks
            if len(chunks) > 1:
                doc_title = f"{base_title} (Part {i+1}/{len(chunks)})"
            else:
                doc_title = base_title
            
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    title=doc_title,
                    url=url,
                    category=category,
                    text=chunk,
                    metadata={
                        "source_type": "crawled",
                        "chunk": i + 1,
                        "total_chunks": len(chunks),
                        "has_products": len(products) > 0,
                        "product_count": len(products),
                    },
                )
            )
        
        return docs
