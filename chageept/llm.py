"""LLM generation layer using HuggingFace inference API (free tier).

Supports Llama-3-8B-Instruct or Mistral-7B-Instruct models.
Falls back to a simple template-based response if LLM unavailable.
"""
import os
from typing import List, Optional
from huggingface_hub import InferenceClient


SYSTEM_PROMPT = """You are a helpful and friendly CHAGEE Philippines assistant.
Use ONLY the provided context to answer questions. Be accurate and factual.

CRITICAL RULES:
- NEVER speculate, guess, or invent information not explicitly in the context
- NEVER say things like "it seems like", "there might be", "it appears that" unless the context explicitly states it
- NEVER suggest there's more information that was "cut off" or incomplete
- If information is partial, only present what you have - don't mention what's missing
- For store locations, only list stores with their exact addresses as given
- For menu items, only list products explicitly named in the context

When listing items (menu, drinks, products, stores):
- List ALL items found in the context with their complete details
- Use bullet points or numbered lists for clarity
- Include descriptions when available

Maintain a warm, premium brand tone. Be helpful and informative."""


class LLMGenerator:
    """LLM generation using HuggingFace Inference API."""

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        api_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.api_token = api_token or os.getenv("HUGGINGFACE_TOKEN")
        self.client = None
        if self.api_token:
            try:
                self.client = InferenceClient(token=self.api_token)
            except Exception:
                self.client = None

    def generate_answer(
        self, query: str, context_chunks: List[str], source_urls: List[str]
    ) -> str:
        """Generate answer from query and retrieved context."""
        if not self.client:
            # Fallback: simple template-based answer
            print("⚠️ LLM unavailable - using fallback template")
            return self._fallback_answer(query, context_chunks, source_urls)

        context_text = "\n\n".join(
            [f"[Context {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)]
        )
        
        # Detect if this is a list query
        list_keywords = ["list", "all", "menu", "items", "what do you have", "what are", "show me"]
        is_list_query = any(kw in query.lower() for kw in list_keywords)
        
        if is_list_query:
            instruction = """List ALL the items/products mentioned in the context.
For each item, include its name and a brief description if available.
Format as a clean numbered or bulleted list."""
        else:
            instruction = "Provide a helpful, conversational answer based on the context above. Be natural and friendly."
        
        user_message = f"""Context from CHAGEE website:
{context_text}

User question: {query}

{instruction}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            # Use more tokens for list queries to prevent cutoffs
            max_tokens = 800 if is_list_query else 400
            
            response = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._fallback_answer(query, context_chunks, source_urls)

    def _fallback_answer(
        self, query: str, context_chunks: List[str], source_urls: List[str]
    ) -> str:
        """Simple template-based fallback when LLM unavailable."""
        if context_chunks:
            # Use the first context chunk as the answer
            answer_text = context_chunks[0]
            # Trim to roughly 100 words for a complete answer
            words = answer_text.split()
            if len(words) > 100:
                answer_text = " ".join(words[:100]) + "..."
            return answer_text
        else:
            return "I don't have that information right now. Please visit the CHAGEE website or contact us for more details."
