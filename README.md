# 🍵 CHAGEEPT

A RAG-powered AI chatbot for CHAGEE Philippines. Ask questions about the menu, store locations, and more!

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-green)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)

## What is this?

CHAGEEPT is a personal fun project - a chatbot that knows everything about CHAGEE Philippines. It uses:

- **RAG (Retrieval-Augmented Generation)** to answer questions based on scraped website data
- **ChromaDB** for vector storage and semantic search
- **HuggingFace LLM** (Llama-3-8B-Instruct) for natural language responses
- **Chainlit** for a ChatGPT-style web interface

## Features

- 🍵 Browse the full CHAGEE menu (Fresh Milk Tea, Brewed Tea, Frappés, etc.)
- 📍 Find store locations in the Philippines
- ℹ️ Learn about CHAGEE's history and brand
- 🔄 Auto-updates database on startup and daily at 3 AM

## Setup

1. **Clone and install dependencies**
   ```bash
   git clone <your-repo>
   cd chageept
   pip install -r requirements.txt
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your HUGGINGFACE_TOKEN
   ```

3. **Build the knowledge base**
   ```bash
   python -m scripts.seed_crawler
   ```

4. **Run the chatbot**
   ```bash
   chainlit run chat_ui.py
   ```

5. Open http://localhost:8000 in your browser 🎉

## Project Structure

```
chageept/
├── chat_ui.py           # Main Chainlit app
├── chageept/
│   ├── scraper.py       # Web scraper for CHAGEE website
│   ├── retriever.py     # ChromaDB vector search
│   ├── llm.py           # HuggingFace LLM integration
│   └── tools.py         # Data models
├── scripts/
│   └── seed_crawler.py  # Database builder
├── public/              # UI assets (logo, favicon)
└── chroma_db/           # Vector database (auto-generated)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Chainlit |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Meta Llama-3-8B-Instruct (via HuggingFace) |
| Scraping | BeautifulSoup4 |

## Deployment

Configured for Railway deployment with Docker. Just push to your repo and connect to Railway.

*This is a personal project and is not affiliated with CHAGEE.*

---
