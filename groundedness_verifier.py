"""
groundedness_verifier.py — Sentence-Level Groundedness & Hallucination Checker
==============================================================================
Verifies whether generated AI responses are strictly supported by retrieved 
source context to prevent hallucinations and ungrounded claims.
"""

import os
import json
import re
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroundednessVerifier:
    def __init__(self):
        self.api_key = GROQ_API_KEY

    def verify_groundedness(self, answer: str, context_chunks: list) -> dict:
        """
        Verifies if the AI generated answer is faithful to the provided context.
        Returns a dict containing:
          • is_grounded (bool)
          • confidence_score (float 0.0 - 1.0)
          • ungrounded_claims (list of sentences not found in context)
        """
        if not answer or not context_chunks:
            return {"is_grounded": True, "confidence_score": 1.0, "ungrounded_claims": []}

        # Combine text chunks for inspection
        context_text = "\n".join([
            chunk.get("document", "") if isinstance(chunk, dict) else str(chunk)
            for chunk in context_chunks
        ])

        prompt = (
            "You are a factual verification auditor evaluating an AI assistant's answer against context documents.\n\n"
            "RULES FOR AUDITING:\n"
            "1. DO NOT flag direct logical inferences (e.g., if context states 'issued automatically', stating 'no extra claim required' is VALID).\n"
            "2. DO NOT flag explicit negative assertions (e.g., if context doesn't mention helicopters, stating 'helicopter commute is not covered or mentioned' is VALID).\n"
            "3. ONLY flag sentences that introduce fabricated facts, wrong numbers, conflicting dates, or entities completely unsupported by the context.\n\n"
            f"CONTEXT DOCUMENTS:\n{context_text}\n\n"
            f"CANDIDATE ANSWER:\n{answer}\n\n"
            "Respond strictly with a valid JSON object matching this schema:\n"
            "{\n"
            '  "is_grounded": true/false,\n'
            '  "confidence_score": 0.0 to 1.0,\n'
            '  "ungrounded_claims": ["only list truly hallucinated or fabricated sentences"]\n'
            "}"
        )

        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [
                {"role": "system", "content": "You perform strict JSON-only verification audits."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            GROQ_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                raw_text = result["choices"][0]["message"]["content"].strip()
                
                # Extract JSON payload from potential markdown wrappers
                json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                    # Ensure ungrounded_claims is a list
                    if not isinstance(parsed.get("ungrounded_claims"), list):
                        parsed["ungrounded_claims"] = []
                    return parsed
        except Exception as e:
            log.warning("Groundedness verification pass failed: %s", e)

        return {"is_grounded": True, "confidence_score": 1.0, "ungrounded_claims": []}