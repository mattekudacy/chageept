"""Chainlit chat interface for CHAGEEPT agent.

Run with: `chainlit run chat_ui.py -w`
"""
import asyncio
import re

import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG components directly (no separate API needed)
from chageept.retriever import SearchTool
from chageept.scraper import ScrapeTool
from chageept.llm import LLMGenerator
from chageept.websearch import TavilySearchTool
from chageept.agent import AgentRunner

GREETING_PATTERN = re.compile(
    r"^\s*(hello|hi|hey|good morning|good afternoon|good evening)\W*\s*$", re.IGNORECASE
)

# Initialize tools (done in the background after the server starts listening,
# so Railway/any host sees an open port immediately instead of timing out
# while the knowledge base builds).
search_tool = None
scrape_tool = None
llm_generator = None
web_search_tool = None
agent_runner = None
tools_ready = asyncio.Event()


def _build_tools():
    """Blocking setup: RAG tools plus an initial crawl if the DB is empty.

    Runs in a worker thread (see on_app_startup) so it never blocks the
    event loop or delays the server from accepting connections.
    """
    global search_tool, scrape_tool, llm_generator, web_search_tool, agent_runner

    print("🚀 Initializing CHAGEEPT...")
    search_tool = SearchTool()
    scrape_tool = ScrapeTool()
    llm_generator = LLMGenerator()
    web_search_tool = TavilySearchTool()
    if web_search_tool.is_available:
        print("🌐 Tavily web search enabled (fallback for CHAGEE-related queries only)")
    else:
        print("🌐 Tavily web search disabled (no TAVILY_API_KEY set)")
    agent_runner = AgentRunner(search_tool, scrape_tool, llm_generator, web_search_tool)

    # Check if database is empty and run initial update
    try:
        doc_count = search_tool.collection.count()
        if doc_count == 0:
            print("📭 Database is empty - running initial data fetch...")
            from scripts.seed_crawler import main as run_crawler
            run_crawler()
            search_tool = SearchTool()  # Reinitialize to pick up new data
            agent_runner = AgentRunner(search_tool, scrape_tool, llm_generator, web_search_tool)
        else:
            print(f"📚 Database loaded with {doc_count} documents")
    except Exception as e:
        print(f"⚠️ Could not check database: {e}")

    print("✅ CHAGEEPT ready!")


@cl.on_app_startup
async def on_app_startup():
    """Kick off tool initialization without blocking server startup."""

    async def run_and_signal():
        await asyncio.to_thread(_build_tools)
        tools_ready.set()

    asyncio.create_task(run_and_signal())


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])


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

    # Handle greetings (exact match, not substring - avoids matching "history", "this", etc.)
    if GREETING_PATTERN.match(user_query):
        await cl.Message(
            content="Hello! 👋 I'm your CHAGEE assistant. I can help you with our menu, store locations, and current promotions. What would you like to know?"
        ).send()
        return

    if not tools_ready.is_set():
        async with cl.Step(name="Warming up CHAGEE knowledge base...") as step:
            await tools_ready.wait()

    history = cl.user_session.get("history", [])

    # Process query through the agent loop (search / scrape / answer)
    async with cl.Step(name="Thinking about CHAGEE knowledge base...") as step:
        try:
            data = agent_runner.run(user_query, history=history)
        except Exception as e:
            print(f"⚠️ Agent run failed: {e}")
            await cl.Message(
                content="❌ Sorry, I encountered an error. Please try again."
            ).send()
            return

    answer = data.get("answer", "No answer available.")
    sources = data.get("sources", [])

    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": answer})
    cl.user_session.set("history", history[-12:])

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
