"""ReAct-style agent loop for CHAGEEPT.

Instead of a single fixed retrieve-then-generate pass, the LLM plans across
multiple steps: it can search the knowledge base (optionally scoped to a
category), scrape a fresh page from the official site when retrieval comes
up empty, and only then produce a final answer. Falls back to a single-shot
RAG pass if the LLM is unavailable.

Uses the OpenAI-compatible `tools` protocol rather than asking the model to
hand-write JSON action objects as plain text: plain-text content with no
tool call IS the final answer, and each tool call's arguments are already
schema-validated JSON by the time we see them.
"""
import json
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .llm import LLMGenerator
from .retriever import SearchTool
from .scraper import ScrapeNotAllowed, ScrapeTool
from .tools import is_list_query
from .websearch import TavilySearchTool

MAX_STEPS = 9
CALL_MODEL_MAX_RETRIES = 3
MAX_SELF_CRITIQUE_RETRIES = 2
# Empirically, off-topic queries against our KB score ~0.48-0.65 cosine
# similarity and genuine on-topic matches score ~0.7-0.8 (gemini-embedding-001,
# our current corpus) - there's no absolute cutoff that cleanly separates the
# two (a generic "menu" query and "starbucks menu" score similarly), so this
# is used only to flag ambiguous results for the planner to judge, not as a
# hard block.
WEAK_MATCH_WARNING_THRESHOLD = 0.68
LOW_THRESHOLD = 0.2
ALLOWED_SCRAPE_DOMAIN = "global.chagee.com"
KNOWN_CATEGORIES = [
    "menu", "stores", "about", "contact", "news",
    "sustainability", "legal", "general",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the CHAGEE knowledge base built from the official CHAGEE website. "
                "ALWAYS try this first, before any other tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text."},
                    "category": {
                        "type": "string",
                        "enum": KNOWN_CATEGORIES,
                        "description": "Set when the question is clearly about one topic; omit otherwise.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scrape",
            "description": (
                "Fetch a fresh page from the official CHAGEE website when the knowledge base has no "
                "good match. Only propose URLs on the global.chagee.com domain."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "A https://global.chagee.com/... URL."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the public web via Tavily. ONLY use this after you have already called "
                "'search' at least once and the knowledge base did not have a good answer. ONLY use "
                "this for questions specifically about CHAGEE (e.g. recent news, current promotions, "
                "prices, store hours not in the knowledge base) - this tool is denied for questions "
                "unrelated to CHAGEE, so decline those in plain text instead of calling it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text for the web search."},
                },
                "required": ["query"],
            },
        },
    },
]

AGENT_SYSTEM_PROMPT = """You are the planning core of a CHAGEE Philippines assistant. You have tools to \
retrieve information from a knowledge base built from the official CHAGEE website, plus a general web search.

The knowledge base is built from the official CHAGEE website, which does NOT contain Philippine prices, \
current promotions, or seasonal/limited-time drinks. For questions about prices, promos, discounts, or \
new/seasonal drinks: run ONE knowledge-base search first, then use web_search - don't retry the knowledge \
base with reworded queries for these topics, since it structurally cannot have this data.

Rules for your final reply (plain text, no tool call):
- Use ONLY information returned by prior tool results plus the conversation history.
- NEVER speculate, guess, or invent information not present in the tool results.
- If nothing relevant was found after searching, say so plainly instead of guessing.
- When listing items (menu, drinks, products, stores), list ALL items found with full detail.
- Maintain a warm, premium brand tone.
- If the user asks something entirely unrelated to CHAGEE (e.g. general trivia, other brands, personal \
advice), politely decline and steer them back to CHAGEE topics instead of calling any tool.
- Call at most one tool per turn."""

CRITIQUE_SYSTEM_PROMPT = """You are a strict quality checker for a CHAGEE Philippines assistant. You will \
be shown the user's question and a draft answer the assistant is about to send. Decide whether the draft \
actually answers the question, or whether it's a premature give-up that should try harder first.

A draft FAILS review (sufficient=false) if:
- It says "I don't have that information" / "I couldn't find" / recommends visiting a store or website,
  WHILE at least one search/web_search action in this conversation has NOT yet been tried with a
  differently-worded query, OR the web_search action has never been used at all despite being available.
- It gives up after only one search attempt or one web_search attempt without trying an alternate phrasing.
- It ignores relevant facts (e.g. a specific price, phone number, or date) that ARE present in the
  observations shown to you, even if buried in an unrelated-looking result.

A draft PASSES review (sufficient=true) if:
- It directly answers the question using facts from the observations, OR
- It has already tried search AND web_search with at least one reworded attempt each, and genuinely
  no source contains the answer, OR
- The question is legitimately unanswerable from any CHAGEE source (unrelated to CHAGEE, or asks for
  private/internal data).

Respond with ONLY a single JSON object, no other text:
{"sufficient": true} or {"sufficient": false, "suggestion": "<a specific, differently-worded action to try next, e.g. a rephrased web_search query>"}"""

GROUNDEDNESS_SYSTEM_PROMPT = """You are a fact-checker for a CHAGEE Philippines assistant. You will be \
shown the tool results gathered so far and a draft answer. Check whether every specific factual claim in \
the draft (a price, a calorie count, an address, a name, a date, a product attribute) is actually \
supported by those tool results - not just topically similar to them.

Common failure to catch: the draft answers about Product A using a fact that the tool results actually \
state about a different, similarly-named Product B (e.g. answering about a "Latte" using a number that \
the source explicitly attributes to a "Milk Tea"). Read carefully which exact product/entity each fact in \
the tool results is attached to.

A draft is grounded (grounded=true) if every specific fact in it is explicitly attached to the same \
product/entity the question asks about in the tool results, or if the draft contains no specific claims \
(pure decline/uncertainty is always grounded).

A draft is NOT grounded (grounded=false) if it states a specific fact that isn't in the tool results at \
all, or attaches a fact from a different product/entity to the one asked about.

Respond with ONLY a single JSON object, no other text:
{"grounded": true} or {"grounded": false, "issue": "<the specific unsupported or misattributed claim>"}"""


class AgentRunner:
    """Runs the search/scrape/final_answer planning loop for a single query."""

    def __init__(
        self,
        search_tool: SearchTool,
        scrape_tool: ScrapeTool,
        llm: LLMGenerator,
        web_search_tool: Optional[TavilySearchTool] = None,
    ):
        self.search_tool = search_tool
        self.scrape_tool = scrape_tool
        self.llm = llm
        self.web_search_tool = web_search_tool or TavilySearchTool()

    def run(self, query: str, history: Optional[List[Dict]] = None) -> Dict:
        if not self.llm.client:
            return self._fallback_rag(query)

        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        messages.extend((history or [])[-6:])
        messages.append({"role": "user", "content": query})

        collected_sources: List[Dict] = []
        seen_urls = set()
        kb_searched = False
        critique_attempts = 0
        groundedness_attempts = 0

        for step in range(MAX_STEPS):
            try:
                message = self._call_model(messages)
            except Exception as e:
                print(f"⚠️ Agent LLM call failed: {e}")
                return self._fallback_rag(query, sources=collected_sources)

            tool_call = message.tool_calls[0] if message.tool_calls else None

            if not tool_call:
                answer = (message.content or "").strip() or "I don't have that information right now."
                if critique_attempts < MAX_SELF_CRITIQUE_RETRIES and step + 1 < MAX_STEPS:
                    verdict = self._critique_answer(query, answer, messages)
                    if not verdict.get("sufficient", True):
                        critique_attempts += 1
                        suggestion = str(verdict.get("suggestion") or "").strip()
                        messages.append({"role": "assistant", "content": answer})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your draft answer was reviewed and rejected as a premature give-up - "
                                "you have not exhausted search/web_search with reworded queries. "
                                f"{('Try this next: ' + suggestion) if suggestion else 'Try a differently-worded search or web_search call before answering.'}"
                            ),
                        })
                        continue
                if groundedness_attempts < MAX_SELF_CRITIQUE_RETRIES and step + 1 < MAX_STEPS:
                    grounded_verdict = self._check_groundedness(answer, messages)
                    if not grounded_verdict.get("grounded", True):
                        groundedness_attempts += 1
                        issue = str(grounded_verdict.get("issue") or "").strip()
                        messages.append({"role": "assistant", "content": answer})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Your draft answer was reviewed and rejected as ungrounded - it stated a "
                                "specific fact not actually supported by the tool results, or attached a "
                                f"fact from a different product/entity. Issue: {issue or 'unspecified'}. "
                                "Correct this - either find the right fact via search/web_search, or say "
                                "plainly that this specific detail isn't available."
                            ),
                        })
                        continue
                return {
                    "answer": answer,
                    "sources": collected_sources,
                    "steps": step + 1,
                }

            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            if name == "search":
                observation, sources, top_score = self._do_search(args)
                if 0 < top_score < WEAK_MATCH_WARNING_THRESHOLD:
                    # Cosine similarity alone can't reliably tell "on-topic
                    # but not covered" apart from "off-topic" (e.g. a generic
                    # "menu" query scores similarly to competitor-brand
                    # queries) - surface it as a hint, not a hard block, so
                    # the planner can judge using the actual retrieved text.
                    observation += (
                        f"\n\n(NOTE: best match scored only {top_score:.2f} - read the results above "
                        "carefully; if they don't actually address the question, try scrape or web_search "
                        "instead of treating this as a good answer.)"
                    )
                kb_searched = True
                for s in sources:
                    if s["url"] not in seen_urls:
                        collected_sources.append(s)
                        seen_urls.add(s["url"])
            elif name == "scrape":
                observation = self._do_scrape(str(args.get("url", "")))
            elif name == "web_search":
                observation, sources = self._do_web_search(args, kb_searched=kb_searched)
                for s in sources:
                    if s["url"] not in seen_urls:
                        collected_sources.append(s)
                        seen_urls.add(s["url"])
            else:
                observation = f"Unknown tool: {name}."

            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {"name": name, "arguments": tool_call.function.arguments},
                }],
            })
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": observation})

        # Ran out of planning steps - force a final answer from the
        # observations already gathered in this conversation (search,
        # scrape, web_search), instead of discarding them and falling back
        # to a fresh KB-only search that ignores web_search results we
        # already have.
        messages.append({
            "role": "user",
            "content": (
                "You are out of steps. Respond now in plain text (no tool call) with the best answer "
                "you can give from the tool results above."
            ),
        })
        try:
            message = self._call_model(messages)
            answer = (message.content or "").strip()
        except Exception:
            answer = ""
        if answer:
            return {"answer": answer, "sources": collected_sources, "steps": MAX_STEPS}

        return self._fallback_rag(query, sources=collected_sources)

    def _render_transcript(self, messages: List[Dict]) -> str:
        """Render the tool-calling conversation (minus the system prompt) as
        readable text, for use in a secondary review LLM call."""
        transcript_lines = []
        for m in messages[1:]:
            role = m["role"].upper()
            if m["role"] == "assistant" and m.get("tool_calls"):
                call = m["tool_calls"][0]
                transcript_lines.append(f"{role} CALLED {call['function']['name']}({call['function']['arguments']})")
            elif m["role"] == "tool":
                transcript_lines.append(f"TOOL RESULT: {m['content']}")
            else:
                transcript_lines.append(f"{role}: {m['content']}")
        return "\n\n".join(transcript_lines)

    def _critique_answer(self, query: str, draft_answer: str, messages: List[Dict]) -> Dict:
        """Ask the model to judge its own draft final answer against the
        original question and the tool calls/results tried so far, so a
        premature give-up gets sent back for another attempt instead of
        reaching the user."""
        transcript = self._render_transcript(messages)
        critique_messages = [
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Conversation so far (tool calls and results):\n{transcript}\n\n"
                    f"Draft answer about to be sent: {draft_answer}"
                ),
            },
        ]
        try:
            message = self._call_model(critique_messages, tools=None)
            raw = (message.content or "").strip()
        except Exception:
            return {"sufficient": True}
        verdict = self._parse_json_object(raw)
        if not isinstance(verdict, dict) or "sufficient" not in verdict:
            return {"sufficient": True}
        return verdict

    def _check_groundedness(self, draft_answer: str, messages: List[Dict]) -> Dict:
        """Ask the model to fact-check its own draft against the tool
        results gathered so far, catching cases where a fact is invented
        outright or borrowed from a similarly-named but different
        product/entity than the one asked about."""
        transcript = self._render_transcript(messages)
        groundedness_messages = [
            {"role": "system", "content": GROUNDEDNESS_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Tool results gathered so far:\n{transcript}\n\n"
                    f"Draft answer to fact-check: {draft_answer}"
                ),
            },
        ]
        try:
            message = self._call_model(groundedness_messages, tools=None)
            raw = (message.content or "").strip()
        except Exception:
            return {"grounded": True}
        verdict = self._parse_json_object(raw)
        if not isinstance(verdict, dict) or "grounded" not in verdict:
            return {"grounded": True}
        return verdict

    def _call_model(self, messages: List[Dict], tools=TOOLS):
        """Call the model with the native tools protocol. Some cloud models
        occasionally return an empty message (no content, no tool_calls) -
        resampling reliably avoids it."""
        message = None
        for _ in range(CALL_MODEL_MAX_RETRIES):
            kwargs = {
                "model": self.llm.model_name,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.2,
            }
            if tools:
                kwargs["tools"] = tools
            response = self.llm.client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            if message.tool_calls or (message.content or "").strip():
                break
        return message

    def _parse_json_object(self, raw: str) -> Optional[Dict]:
        """Parse a JSON object from text that may be wrapped in markdown
        fences (some models add ```json ... ``` even when asked not to)."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def _do_search(self, args: Dict) -> Tuple[str, List[Dict], float]:
        query = str(args.get("query") or "").strip()
        if not query:
            return "Search failed: no query provided.", [], 0.0

        category = args.get("category")
        if category and category not in KNOWN_CATEGORIES:
            category = None

        top_k = 12 if is_list_query(query) else 6
        results = self.search_tool.search(query, top_k=top_k, category=category)
        top_score = results[0].score if results else 0.0
        observation, sources = self._format_search_observation(results)
        return observation, sources, top_score

    def _format_search_observation(self, results) -> Tuple[str, List[Dict]]:
        if not results:
            return "No matching documents found in the knowledge base.", []

        chunks = []
        sources = []
        seen_texts = set()
        for r in results[:10]:
            text_key = r.document.text[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            chunks.append(f"(relevance={r.score:.2f}) {r.document.text[:1200]}")
            title = r.document.title.split(" (Part ")[0]
            sources.append({"title": title, "url": r.document.url})

        return "\n---\n".join(chunks), sources

    def _do_web_search(self, args: Dict, kb_searched: bool) -> Tuple[str, List[Dict]]:
        if not self.web_search_tool.is_available:
            return "Web search unavailable: no Tavily API key configured.", []

        if not kb_searched:
            return (
                "Web search denied: you must call 'search' first before falling back to the web.",
                [],
            )

        query = str(args.get("query") or "").strip()
        if not query:
            return "Web search failed: no query provided.", []

        try:
            results = self.web_search_tool.search(query, max_results=5)
        except Exception as e:
            return f"Web search failed: {e}", []

        if not results:
            return "Web search returned no results.", []

        chunks = []
        sources = []
        for r in results:
            content = (r.get("content") or "")[:1200]
            chunks.append(f"[{r.get('title', '')}]: {content}")
            sources.append({"title": r.get("title", "Web result"), "url": r.get("url", "")})

        return "\n---\n".join(chunks), sources

    def _do_scrape(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc != ALLOWED_SCRAPE_DOMAIN:
            return f"Scrape denied: {url or '(empty)'} is outside the allowed domain ({ALLOWED_SCRAPE_DOMAIN})."
        try:
            docs = self.scrape_tool.scrape(url)
        except ScrapeNotAllowed:
            return f"Scrape denied by robots.txt: {url}"
        except Exception as e:
            return f"Scrape failed: {e}"

        if docs:
            self.search_tool.add_documents(docs)
            return f"Scraped and indexed {len(docs)} new sections from {url}. Search again to use them."
        return f"Scraped {url} but found no usable content."

    def _fallback_rag(self, query: str, sources: Optional[List[Dict]] = None) -> Dict:
        """Single-shot retrieve-then-generate, used when the LLM is unavailable
        or fails to produce an answer through the tool-calling loop."""
        is_list = is_list_query(query)
        top_k = 12 if is_list else 6
        results = self.search_tool.search(query, top_k=top_k)
        top_score = results[0].score if results else 0.0

        context_chunks = []
        seen_texts = set()
        for r in results[: (10 if is_list else 5)]:
            text_key = r.document.text[:100]
            if text_key not in seen_texts:
                context_chunks.append(r.document.text[:1500])
                seen_texts.add(text_key)

        result_sources = list(sources or [])
        seen_urls = {s["url"] for s in result_sources}
        for r in results[:6]:
            if r.document.url not in seen_urls:
                title = r.document.title.split(" (Part ")[0]
                result_sources.append({"title": title, "url": r.document.url})
                seen_urls.add(r.document.url)

        if context_chunks and top_score > LOW_THRESHOLD:
            answer = self.llm.generate_answer(query, context_chunks)
        else:
            answer = "I couldn't find specific information about that. Please visit the CHAGEE website for more details."

        return {"answer": answer, "sources": result_sources, "steps": 0}
