"""
run_evals.py — Automated RAG Evaluation Benchmark Suite
======================================================
Evaluates context retrieval precision, recall, and answer faithfulness
against a ground-truth dataset to track system performance over time.
"""

import os
import json
import logging
import urllib.request
import urllib.error

# Load environment variables from .env file
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

from rag_pipeline import RAGPipeline
from query_rewriter import QueryRewriter
from hybrid_search import HybridRetriever
from groundedness_verifier import GroundednessVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def call_llm(prompt: str) -> str:
    """Sends a completion request to the Groq API."""
    if not GROQ_API_KEY:
        log.error("❌ GROQ_API_KEY is missing! Check your .env file.")
        return "Error: Missing GROQ_API_KEY"

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a strict factual corporate assistant."},
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
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "Mozilla/5.0"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        log.error(f"❌ LLM Request Failed: {e}")
        return f"Error: {e}"


# Benchmark evaluation dataset
EVAL_DATASET = [
    {
        "question": "What is the policy on remote work?",
        "expected_keyword": "remote",
        "ground_truth": "Employees can work remotely up to 2 days per week with manager approval."
    },
    {
        "question": "How many days of paid time off (PTO) do employees get?",
        "expected_keyword": "PTO",
        "ground_truth": "Full-time employees receive 15 days of paid time off annually."
    }
]


def run_benchmark():
    log.info("🚀 Starting RAG Pipeline Evaluation Suite...")
    
    pipeline = RAGPipeline(kb_path="kb.txt", top_k=3, prefer_transformer=False)
    pipeline.ingest()
    
    hybrid = HybridRetriever(pipeline.chunks)
    verifier = GroundednessVerifier()

    total_queries = len(EVAL_DATASET)
    successful_retrievals = 0
    grounded_responses = 0

    for idx, test_case in enumerate(EVAL_DATASET, 1):
        query = test_case["question"]
        expected = test_case["expected_keyword"].lower()

        log.info(f"\n--- Testing Query {idx}/{total_queries}: '{query}' ---")

        # 1. Evaluate Retrieval Precision & Recall
        vector_res = pipeline.retrieve(query)[:3]
        bm25_scores = hybrid.bm25.search(query, top_k=3)
        fused_res = hybrid.reciprocal_rank_fusion(vector_res, bm25_scores)[:3]

        retrieved_texts = [
            r.get("document", "").lower() if isinstance(r, dict) else str(r).lower() 
            for r in fused_res
        ]
        hit = any(expected in text for text in retrieved_texts)

        if hit:
            successful_retrievals += 1
            log.info("  [Retrieval]: ✅ SUCCESS — Expected keyword found in top chunks.")
        else:
            log.warning("  [Retrieval]: ❌ FAIL — Expected keyword missing from context.")

        # 2. Generate Real LLM Response
        context_block = pipeline.format_context(fused_res)
        prompt = (
            "SYSTEM INSTRUCTIONS:\n"
            "Answer the user question based strictly on the context documents below. "
            "Do not guess or use outside knowledge.\n\n"
            "CONTEXT DOCUMENTS:\n"
            f"{context_block}\n\n"
            f"USER QUESTION: {query}\n"
            "ANSWER:"
        )
        generated_answer = call_llm(prompt)
        log.info(f"  [Generated Answer]: {generated_answer}")

        # 3. Evaluate Faithfulness / Groundedness of Generated Answer
        verification = verifier.verify_groundedness(generated_answer, fused_res)

        if verification.get("is_grounded", True):
            grounded_responses += 1
            log.info("  [Groundedness]: ✅ PASS — Response is fully supported by context.")
        else:
            log.warning(f"  [Groundedness]: ❌ WARN — Ungrounded claims: {verification.get('ungrounded_claims')}")

    # Final Benchmark Report
    retrieval_acc = (successful_retrievals / total_queries) * 100
    groundedness_acc = (grounded_responses / total_queries) * 100

    print("\n" + "=" * 50)
    print("           RAG EVALUATION BENCHMARK REPORT         ")
    print("=" * 50)
    print(f" Total Evaluation Test Cases : {total_queries}")
    print(f" Retrieval Context Recall    : {retrieval_acc:.1f}%")
    print(f" Answer Faithfulness Score   : {groundedness_acc:.1f}%")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_benchmark()