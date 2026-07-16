"""ReAct-style agent loop for CHAGEEPT.

Instead of a single fixed retrieve-then-generate pass, the LLM plans across
multiple steps: it can search the knowledge base (optionally scoped to a
category), scrape a fresh page from the official site when retrieval comes
up empty, and only then produce a final answer. Falls back to a single-shot
RAG pass if the LLM is unavailable or doesn't follow the JSON protocol.
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .llm import LLMGenerator
from .retriever import SearchTool
from .scraper import ScrapeNotAllowed, ScrapeTool
from .websearch import TavilySearchTool

MAX_STEPS = 9
CALL_MODEL_MAX_RETRIES = 3
MAX_SELF_CRITIQUE_RETRIES = 2
LOW_THRESHOLD = 0.2
ALLOWED_SCRAPE_DOMAIN = "global.chagee.com"
KNOWN_CATEGORIES = {
    "menu", "stores", "about", "contact", "news",
    "sustainability", "legal", "general",
}
LIST_KEYWORDS = [
    "list", "all", "menu", "items", "drinks", "what do you have",
    "what are", "show me", "options", "available",
]
CHAGEE_TOPIC_KEYWORDS = [
    "chagee",
]

AGENT_SYSTEM_PROMPT = """You are the planning core of a CHAGEE Philippines assistant. You have tools to \
retrieve information from a knowledge base built from the official CHAGEE website, plus a general web \
search as a last resort. Think step by step, then respond with ONLY a single JSON object - no other \
text, no markdown fences.

Available actions:
1. {"action": "search", "query": "<search text>", "category": "<menu|stores|about|contact|news|sustainability|legal|general|null>"}
   Searches the knowledge base. Set category when the question is clearly about one topic, else null.
   ALWAYS try this first, before any other action.
2. {"action": "scrape", "url": "<https://global.chagee.com/... URL>"}
   Fetches a fresh page from the official CHAGEE website when the knowledge base has no good match.
   Only propose URLs on the global.chagee.com domain.
3. {"action": "web_search", "query": "<search text, must mention CHAGEE>"}
   Searches the public web via Tavily. ONLY use this after you have already run "search" at least once
   and the knowledge base did not have a good answer. ONLY use this for questions specifically about
   CHAGEE (e.g. recent news, current promotions, store hours not in the knowledge base). NEVER use this
   for questions unrelated to CHAGEE - refuse those in your final_answer instead. The query you send MUST
   explicitly mention "CHAGEE".
4. {"action": "final_answer", "answer": "<your reply to the user>"}
   Ends the process and returns your answer.

Rules for final_answer:
- Use ONLY information returned by prior search/scrape/web_search observations plus the conversation history.
- NEVER speculate, guess, or invent information not present in the observations.
- If nothing relevant was found after searching, say so plainly instead of guessing.
- When listing items (menu, drinks, products, stores), list ALL items found with full detail.
- Maintain a warm, premium brand tone.
- If the user asks something entirely unrelated to CHAGEE (e.g. general trivia, other brands, personal \
advice), politely decline and steer them back to CHAGEE topics instead of using any tool.
- Take exactly one action per turn."""

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

        for step in range(MAX_STEPS):
            try:
                raw = self._call_model(messages)
            except Exception as e:
                print(f"⚠️ Agent LLM call failed: {e}")
                return self._fallback_rag(query, sources=collected_sources)

            action = self._parse_action(raw)
            if not action or "action" not in action:
                # Model broke the JSON protocol (e.g. plain narration instead
                # of an action) - nudge it to retry rather than surfacing the
                # raw text as if it were the answer.
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": (
                        "Observation: Invalid response - you must reply with ONLY a single JSON "
                        "action object (search, scrape, web_search, or final_answer), no other text."
                    ),
                })
                continue

            kind = action.get("action")

            if kind == "final_answer":
                answer = str(action.get("answer", "")).strip() or "I don't have that information right now."
                if critique_attempts < MAX_SELF_CRITIQUE_RETRIES and step + 1 < MAX_STEPS:
                    verdict = self._critique_answer(query, answer, messages)
                    if not verdict.get("sufficient", True):
                        critique_attempts += 1
                        suggestion = str(verdict.get("suggestion") or "").strip()
                        messages.append({"role": "assistant", "content": raw})
                        messages.append({
                            "role": "user",
                            "content": (
                                "Observation: Your draft answer was reviewed and rejected as a premature "
                                "give-up - you have not exhausted search/web_search with reworded queries. "
                                f"{('Try this next: ' + suggestion) if suggestion else 'Try a differently-worded search or web_search query before answering.'}"
                            ),
                        })
                        continue
                return {
                    "answer": answer,
                    "sources": collected_sources,
                    "steps": step + 1,
                }

            if kind == "search":
                observation, sources, _ = self._do_search(action)
                kb_searched = True
                for s in sources:
                    if s["url"] not in seen_urls:
                        collected_sources.append(s)
                        seen_urls.add(s["url"])
            elif kind == "scrape":
                observation = self._do_scrape(str(action.get("url", "")))
            elif kind == "web_search":
                observation, sources = self._do_web_search(action, kb_searched=kb_searched)
                for s in sources:
                    if s["url"] not in seen_urls:
                        collected_sources.append(s)
                        seen_urls.add(s["url"])
            else:
                observation = "Unknown action. Use search, scrape, web_search, or final_answer."

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        # Ran out of planning steps - force a final answer from the
        # observations already gathered in this conversation (search,
        # scrape, web_search), instead of discarding them and falling back
        # to a fresh KB-only search that ignores web_search results we
        # already have.
        messages.append({
            "role": "user",
            "content": (
                "Observation: You are out of steps. Respond now with ONLY "
                '{"action": "final_answer", "answer": "..."} using the best '
                "answer you can give from the observations above."
            ),
        })
        try:
            raw = self._call_model(messages)
            action = self._parse_action(raw)
        except Exception:
            action = None
        if action and action.get("action") == "final_answer":
            answer = str(action.get("answer", "")).strip()
            if answer:
                return {"answer": answer, "sources": collected_sources, "steps": MAX_STEPS}

        return self._fallback_rag(query, sources=collected_sources)

    def _critique_answer(self, query: str, draft_answer: str, messages: List[Dict]) -> Dict:
        """Ask the model to judge its own draft final_answer against the
        original question and the actions/observations tried so far, so a
        premature give-up gets sent back for another attempt instead of
        reaching the user."""
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages[1:]
        )
        critique_messages = [
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original question: {query}\n\n"
                    f"Conversation so far (actions and observations):\n{transcript}\n\n"
                    f"Draft answer about to be sent: {draft_answer}"
                ),
            },
        ]
        try:
            raw = self._call_model(critique_messages)
        except Exception:
            return {"sufficient": True}
        verdict = self._parse_action(raw)
        if not isinstance(verdict, dict) or "sufficient" not in verdict:
            return {"sufficient": True}
        return verdict

    def _call_model(self, messages: List[Dict]) -> str:
        # Some cloud models occasionally emit a spontaneous, unrequested tool
        # call instead of the JSON action we asked for, leaving content
        # empty. Resampling reliably avoids it.
        content = ""
        for _ in range(CALL_MODEL_MAX_RETRIES):
            response = self.llm.client.chat.completions.create(
                model=self.llm.model_name,
                messages=messages,
                max_tokens=2000,
                temperature=0.2,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                break
        return content

    def _parse_action(self, raw: str) -> Optional[Dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # The model sometimes appends a hallucinated observation/next-action
        # after its first JSON object instead of stopping. Only take the
        # first balanced {...} span, not a greedy match to the last "}" in
        # the whole response.
        first_brace = raw.find("{")
        if first_brace == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(first_brace, len(raw)):
            char = raw[i]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[first_brace : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _do_search(self, action: Dict) -> Tuple[str, List[Dict], float]:
        query = str(action.get("query") or "").strip()
        if not query:
            return "Search failed: no query provided.", [], 0.0

        category = action.get("category")
        if category:
            category = str(category).strip().lower()
            if category not in KNOWN_CATEGORIES:
                category = None

        top_k = 12 if self._is_list_query(query) else 6
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

    def _do_web_search(self, action: Dict, kb_searched: bool) -> Tuple[str, List[Dict]]:
        if not self.web_search_tool.is_available:
            return "Web search unavailable: no Tavily API key configured.", []

        if not kb_searched:
            return (
                "Web search denied: you must search the knowledge base first. "
                "Try the 'search' action before falling back to the web.",
                [],
            )

        query = str(action.get("query") or "").strip()
        if not query:
            return "Web search failed: no query provided.", []

        if not self._mentions_chagee(query):
            return (
                "Web search denied: this tool is restricted to CHAGEE-related "
                "queries. Rephrase the query to explicitly mention CHAGEE, or "
                "if the user's question isn't about CHAGEE, decline it in your final_answer instead.",
                [],
            )

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

    def _mentions_chagee(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in CHAGEE_TOPIC_KEYWORDS)

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

    def _is_list_query(self, query: str) -> bool:
        query_lower = query.lower()
        return any(kw in query_lower for kw in LIST_KEYWORDS)

    def _fallback_rag(self, query: str, sources: Optional[List[Dict]] = None) -> Dict:
        """Single-shot retrieve-then-generate, used when the LLM is unavailable
        or fails to follow the tool protocol."""
        is_list = self._is_list_query(query)
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
