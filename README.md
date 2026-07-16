# 🍵 CHAGEEPT

A RAG-powered AI chatbot for CHAGEE Philippines. Ask questions about the menu, store locations, and more!

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Chainlit](https://img.shields.io/badge/UI-Chainlit-green)
![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-orange)

## What is this?

CHAGEEPT is a personal fun project - a chatbot that knows everything about CHAGEE Philippines. It uses:

- **RAG (Retrieval-Augmented Generation)** to answer questions based on scraped website data
- **Qdrant Cloud** for vector storage and semantic search
- **Ollama Cloud LLM** (via Ollama's OpenAI-compatible API) for natural language responses
- **Gemini embeddings** (via Google AI Studio's OpenAI-compatible API) for vector search
- **Agentic tool-calling loop** - the LLM plans across knowledge-base search, on-demand scraping, and a
  Tavily web-search fallback (CHAGEE-only, used only after the knowledge base comes up short)
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
   # Edit .env and add your GEMINI_API_KEY (embeddings), OLLAMA_API_KEY (chat),
   # and QDRANT_URL/QDRANT_API_KEY (vector storage)
   # Optionally add TAVILY_API_KEY to enable web-search fallback
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
│   ├── agent.py         # Tool-calling planning loop (search/scrape/web_search/answer)
│   ├── scraper.py       # Web scraper for CHAGEE website
│   ├── retriever.py     # Qdrant vector search
│   ├── llm.py           # Ollama Cloud LLM integration
│   ├── websearch.py     # Tavily web search fallback
│   └── tools.py         # Data models
├── scripts/
│   └── seed_crawler.py  # Database builder
└── public/              # UI assets (logo, favicon)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Chainlit |
| Vector DB | Qdrant Cloud |
| Embeddings | gemini-embedding-001 (via Google AI Studio) |
| LLM | gpt-oss:120b-cloud (via Ollama Cloud) |
| Scraping | BeautifulSoup4 |

## Deployment

Configured for Railway deployment with Docker. Just push to your repo and connect to Railway.

---

*This is a personal project and is not affiliated with CHAGEE.*
