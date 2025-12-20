"""Chainlit chat interface for CHAGEEPT agent.

Run with: `chainlit run chat_ui.py -w`
"""
import chainlit as cl
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG components directly (no separate API needed)
from chageept.retriever import SearchTool
from chageept.scraper import ScrapeTool, ScrapeNotAllowed
from chageept.llm import LLMGenerator

# Configuration
HIGH_THRESHOLD = 0.78
LOW_THRESHOLD = 0.3

# Initialize tools (will be done on startup)
search_tool = None
scrape_tool = None
llm_generator = None


def initialize_tools():
    """Initialize RAG tools and run initial data fetch if needed."""
    global search_tool, scrape_tool, llm_generator
    
    print("🚀 Initializing CHAGEEPT...")
    search_tool = SearchTool()
    scrape_tool = ScrapeTool()
    llm_generator = LLMGenerator()
    
    # Check if database is empty and run initial update
    try:
        doc_count = search_tool.collection.count()
        if doc_count == 0:
            print("📭 Database is empty - running initial data fetch...")
            from scripts.seed_crawler import main as run_crawler
            run_crawler()
            search_tool = SearchTool()  # Reinitialize to pick up new data
        else:
            print(f"📚 Database loaded with {doc_count} documents")
    except Exception as e:
        print(f"⚠️ Could not check database: {e}")
    
    print("✅ CHAGEEPT ready!")


# Initialize on module load
initialize_tools()


def process_query(query: str, candidate_urls: List[str] = None):
    """Process a query through the RAG pipeline."""
    candidate_urls = candidate_urls or []
    
    # Search vector DB
    results = search_tool.search(query, top_k=4)
    used_scrape = False
    top_score = results[0].score if results else 0.0
    
    if top_score >= HIGH_THRESHOLD:
        pass  # High confidence - use retrieved chunks
    elif top_score >= LOW_THRESHOLD:
        # Medium confidence - try on-demand scraping
        scraped_docs = []
        for url in candidate_urls:
            try:
                docs = scrape_tool.scrape(url)
                scraped_docs.extend(docs)
            except (ScrapeNotAllowed, Exception):
                continue
        if scraped_docs:
            search_tool.add_documents(scraped_docs)
            results = search_tool.search(query, top_k=4)
            used_scrape = True
            top_score = results[0].score if results else 0.0

    # Extract context and sources
    context_chunks = [r.snippet for r in results[:3]]
    source_urls = []
    sources = []
    seen = set()
    for r in results[:2]:
        if r.document.url not in seen:
            sources.append({"title": r.document.title, "url": r.document.url})
            source_urls.append(r.document.url)
            seen.add(r.document.url)

    # Generate answer
    if context_chunks and top_score > 0.2:
        answer = llm_generator.generate_answer(query, context_chunks, source_urls)
    else:
        answer = "I couldn't find specific information about that. Please visit the CHAGEE website for more details."

    return {
        "answer": answer,
        "sources": sources,
        "used_scrape": used_scrape,
        "confidence_score": top_score
    }


@cl.on_chat_start
async def on_chat_start():
    pass


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="🍵 What drinks do you have?",
            message="What drinks and beverages does CHAGEE offer?",
        ),
        cl.Starter(
            label="📍 Find nearest store",
            message="Where are CHAGEE stores located in Metro Manila?",
        ),
        cl.Starter(
            label="🎉 Current promotions",
            message="What promotions and deals are available this month?",
        ),
        cl.Starter(
            label="💡 What is CHAGEE?",
            message="Tell me about CHAGEE Philippines",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content.strip()
    if not user_query:
        await cl.Message(content="Please ask me a question!").send()
        return

    # Handle greetings
    greetings = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
    if any(g in user_query.lower() for g in greetings) and len(user_query.split()) <= 3:
        await cl.Message(
            content="Hello! 👋 I'm your CHAGEE assistant. I can help you with our menu, store locations, and current promotions. What would you like to know?"
        ).send()
        return

    # Process query through RAG pipeline
    async with cl.Step(name="Searching CHAGEE knowledge base...") as step:
        try:
            data = process_query(user_query)
        except Exception as e:
            await cl.Message(
                content=f"❌ Sorry, I encountered an error. Please try again."
            ).send()
            return

    answer = data.get("answer", "No answer available.")
    sources = data.get("sources", [])

    # Format response
    response_parts = [answer]

    if sources:
        response_parts.append("\n\n**Sources:**")
        for src in sources:
            response_parts.append(f"• [{src['title']}]({src['url']})")

    # Action buttons
    actions = [
        cl.Action(
            name="view_menu",
            value="menu",
            payload={"url": "https://global.chagee.com/ph/en/menu"},
            label="🍵 View Menu",
            description="Browse CHAGEE menu",
        ),
        cl.Action(
            name="find_store",
            value="store",
            payload={"url": "https://global.chagee.com/ph/en/stores"},
            label="📍 Find Store",
            description="Locate nearest CHAGEE store",
        ),
        cl.Action(
            name="promotions",
            value="promo",
            payload={"url": "https://global.chagee.com/ph/en/promotions"},
            label="🎉 Check Promotions",
            description="See current offers",
        ),
    ]

    await cl.Message(content="\n".join(response_parts), actions=actions).send()


@cl.action_callback("view_menu")
async def on_action_menu(action: cl.Action):
    await cl.Message(
        content="🍵 [View CHAGEE Menu](https://global.chagee.com/ph/en/product)"
    ).send()


@cl.action_callback("find_store")
async def on_action_store(action: cl.Action):
    await cl.Message(
        content="📍 [Find CHAGEE Stores](https://global.chagee.com/ph/en/stores)"
    ).send()


@cl.action_callback("promotions")
async def on_action_promo(action: cl.Action):
    await cl.Message(
        content="🎉 [Check Promotions](https://global.chagee.com/ph/en/media-centre)"
    ).send()
