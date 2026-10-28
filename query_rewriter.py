"""
query_rewriter.py — Conversational Query Rewriter & Expansion Engine
=====================================================================
Transforms multi-turn follow-up queries into fully explicit, standalone 
search queries before vector retrieval using conversation context.
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class QueryRewriter:
    def __init__(self):
        self.api_key = GROQ_API_KEY

    def rewrite_query(self, user_query: str, history_str: str) -> str:
        """
        Rewrites a user follow-up query using history so it can stand alone for vector search.
        If history is empty or query is already explicit, returns the original query.
        """
        if not history_str or "None (" in history_str:
            return user_query

        prompt = (
            "You are an expert search query reformulator.\n"
            "Given the conversation history and a follow-up user question, rewrite the follow-up question "
            "into a clear, self-contained search query suitable for document retrieval.\n"
            "DO NOT answer the question. ONLY return the rewritten standalone question.\n\n"
            f"CONVERSATION HISTORY:\n{history_str}\n\n"
            f"USER FOLLOW-UP QUESTION: {user_query}\n\n"
            "STANDALONE SEARCH QUERY:"
        )

        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": "You reformulate search queries strictly and concisely."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        data = json.dumps(payload).encode("utf-8")

        api_key = os.environ.get("GROQ_API_KEY", self.api_key)
        req = urllib.request.Request(
            GROQ_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                rewritten = result["choices"][0]["message"]["content"].strip()
                rewritten = re.sub(r"[\u202f\xa0\s]+", " ", rewritten)
                log.info("Original Query: '%s' ➔ Rewritten Query: '%s'", user_query, rewritten)
                return rewritten
        except Exception as e:
            log.warning("Query rewriting failed: %s. Falling back to original query.", e)
            return user_query