"""Example seed crawler that uses the ScrapeTool and adds docs to SearchTool.

Run as: `python -m scripts.seed_crawler` (from project root)
"""
from chageept.scraper import ScrapeTool
from chageept.retriever import SearchTool


SEED_URLS = [
    # Add the canonical CHAGEE pages you want seeded for indexing
    "https://global.chagee.com/ph/en/product",
    "https://global.chagee.com/ph/en/about",
    "https://global.chagee.com/ph/en/stores",
    "https://global.chagee.com/ph/en/contact"
]


def main():
    scraper = ScrapeTool()
    # Use same persist directory as API
    retriever = SearchTool(persist_directory="./chroma_db")
    
    total_docs = 0
    for u in SEED_URLS:
        try:
            docs = scraper.scrape(u, throttle_seconds=1.0)
            retriever.add_documents(docs)
            total_docs += len(docs)
            print(f"✓ Indexed {len(docs)} chunks from {u}")
        except Exception as e:
            print(f"✗ Failed to scrape {u}: {e}")

    print(f"\n🎉 Seed crawl complete. Total documents indexed: {total_docs}")
    print("Vector DB persisted at: ./chroma_db")


if __name__ == "__main__":
    main()
