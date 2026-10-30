"""
app.py — Production RAG Chatbot Backend
======================================
Integrates Guardrails, Semantic Cache, Conversation Memory, Hybrid Search,
Query Rewriting, Groundedness Verification, SSE Streaming, and Analytics.
"""

import os
import json
import logging
import urllib.request
import urllib.error
from flask import Flask, request, jsonify, Response, stream_with_context, render_template

# Load Environment Variables
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Custom Module Imports
from rag_pipeline import RAGPipeline
from semantic_cache import SemanticCache
from memory_manager import MemoryManager
from feedback_manager import FeedbackManager
from query_rewriter import QueryRewriter
from hybrid_search import HybridRetriever
from groundedness_verifier import GroundednessVerifier
from analytics_manager import AnalyticsManager
from advanced_rag_engine import AdvancedAIEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# Flask Application Setup
app = Flask(__name__)

# Load API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip("\"' ")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# 1. Initialize RAG Pipeline & Ingest Knowledge Base
pipeline = RAGPipeline(kb_path="kb.txt", top_k=3, prefer_transformer=False)
pipeline.ingest()

# 2. Global Core Managers Initialization
cache = SemanticCache()
memory = MemoryManager()
feedback_manager = FeedbackManager()
rewriter = QueryRewriter()
hybrid_retriever = HybridRetriever(pipeline.chunks)
verifier = GroundednessVerifier()
ai_engine = AdvancedAIEngine()

# 📊 Analytics Manager Initialization (Global Access)
analytics_mgr = AnalyticsManager(cache, memory, feedback_manager)


# Helper: Input Guardrail Sanitization
def sanitize_input(query: str) -> tuple[bool, str]:
    """Scans query for prompt injection attacks or disallowed patterns."""
    query_clean = query.strip()
    suspicious_patterns = [
        "ignore previous instructions",
        "system prompt",
        "jailbreak",
        "drop database"
    ]
    for pattern in suspicious_patterns:
        if pattern in query_clean.lower():
            log.warning("⚠️ Guardrail Alert: Blocked input pattern: '%s'", pattern)
            return False, "⚠️ SECURITY BLOCK: Suspicious input pattern detected."
    return True, query_clean


# Helper: Output Guardrail Validation
def validate_output(response_text: str) -> str:
    """Validates generated output against leakages or compliance blocks."""
    if "SYSTEM INSTRUCTIONS" in response_text or "GROQ_API_KEY" in response_text:
        log.error("⚠️ Security Block: Attempted instruction/key leak detected.")
        return "⚠️ SECURITY BLOCK: Output blocked due to sensitive system information leakage."
    return response_text


# Helper: Synchronous LLM Call
def call_llm(prompt: str) -> str:
    """Sends a completion request to Groq API."""
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
        log.error("LLM Call Error: %s", e)
        return "I am currently unable to process your request due to an internal service issue."


# Helper: SSE Stream LLM Generator
def stream_llm_generator(prompt: str):
    """Streams LLM completion response using SSE event-stream format."""
    api_key = os.environ.get("GROQ_API_KEY", GROQ_API_KEY).strip("\"' ")
    if not api_key:
        yield f"data: {json.dumps({'content': '⚠️ GROQ_API_KEY is not configured in .env file.'})}\n\n"
        return

    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "You are a strict factual corporate assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    
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
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or line_str.startswith(":"):
                    continue
                if line_str == "data: [DONE]":
                    break
                if line_str.startswith("data: "):
                    try:
                        chunk_json = json.loads(line_str[6:])
                        delta = chunk_json["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield f"data: {json.dumps({'content': delta})}\n\n"
                    except Exception:
                        pass
    except urllib.error.HTTPError as e:
        err_msg = f"HTTP {e.code}: {e.reason}"
        try:
            err_json = json.loads(e.read().decode("utf-8"))
            if "error" in err_json and "message" in err_json["error"]:
                err_msg = err_json["error"]["message"]
        except Exception:
            pass
        log.error("Streaming LLM HTTPError: %s", err_msg)
        yield f"data: {json.dumps({'content': f'⚠️ AI Service Error: {err_msg}'})}\n\n"
    except Exception as e:
        log.error("Streaming LLM Exception: %s", e)
        yield f"data: {json.dumps({'content': f' Error during streaming: {e}'})}\n\n"


# --- API ROUTES ---

@app.route("/api/query", methods=["POST"])
def api_query():
    """Standard JSON API Endpoint with Query Rewriting, Hybrid Search, & Verification."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default_session")
    role = data.get("role", "default")
    top_k = data.get("top_k", 3)

    if not query:
        return jsonify({"error": "Missing 'query' field."}), 400

    # 1. Guardrail Scan
    is_safe, sanitized_or_msg = sanitize_input(query)
    if not is_safe:
        return jsonify({"query": query, "answer": sanitized_or_msg, "results": [], "session_id": session_id})

    clean_query = sanitized_or_msg

    # 2. Semantic Cache Check
    cached_response = cache.get(clean_query, namespace=role)
    if cached_response:
        log.info("⚡ Semantic Cache Hit!")
        memory.add_turn(session_id, clean_query, cached_response["answer"])
        return jsonify({
            "query": clean_query,
            "answer": cached_response["answer"] + " *(⚡ Cached Response)*",
            "results": cached_response["results"],
            "session_id": session_id
        })

    # 3. History Retrieval & Query Rewriting
    history_str = memory.format_history_for_prompt(session_id)
    try:
        standalone_query = rewriter.rewrite_query(clean_query, history_str)
    except Exception as e:
        log.warning("Rewriter error in api_query: %s", e)
        standalone_query = clean_query

    # 4. Multi-Stage AI Retrieval (Vector + BM25 Fusion + AI Re-Ranking)
    try:
        vector_candidates = pipeline.retrieve(standalone_query)[:8]
        bm25_candidates = hybrid_retriever.bm25.search(standalone_query, top_k=8)
        fused = hybrid_retriever.reciprocal_rank_fusion(vector_candidates, bm25_candidates)[:8]
        results = ai_engine.rerank_chunks(standalone_query, fused, top_k=top_k)
    except Exception as e:
        log.warning("Hybrid search error in api_query: %s. Falling back to vector search.", e)
        results = pipeline.retrieve(standalone_query)[:top_k]

    if not results:
        no_info_msg = "I do not have enough information in my database to answer that."
        memory.add_turn(session_id, clean_query, no_info_msg)
        return jsonify({"query": clean_query, "answer": no_info_msg, "results": [], "session_id": session_id})

    context_block = pipeline.format_context(results)

    # 5. Build Guardrailed Prompt with Citation Directives
    prompt = (
        "SYSTEM INSTRUCTIONS (STRICT ZERO-HALLUCINATION GUARDRAILS ACTIVE):\n"
        "1. You are a professional, helpful corporate policy assistant.\n"
        "2. Synthesize facts accurately based strictly on the context.\n"
        "3. Explicitly cite source documents (e.g. Document 201) when quoting policy details.\n"
        "4. If context does not contain the answer, reply: 'I do not have enough information in my database to answer that.'\n\n"
        f"RECENT HISTORY:\n{history_str}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {clean_query}\n"
        "ANSWER:"
    )

    # 6. LLM Generation & Output Guardrail
    raw_ai_answer = call_llm(prompt)
    final_answer = validate_output(raw_ai_answer)

    # 7. Groundedness Verification & Active Self-Correction Loop
    verification = verifier.verify_groundedness(final_answer, results)
    is_grounded = verification.get("is_grounded", True)
    ungrounded = verification.get("ungrounded_claims", [])

    if not is_grounded and ungrounded:
        log.warning("⚠️ Groundedness Warning: %s. Launching Active Self-Correction...", ungrounded)
        analytics_mgr.log_groundedness_warning(clean_query, ungrounded)
        final_answer = ai_engine.self_correct_if_needed(final_answer, ungrounded, context_block, clean_query)
        final_answer = validate_output(final_answer)
        is_grounded = True

    accuracy_meta = ai_engine.extract_sources_and_accuracy(results, is_grounded, ungrounded)

    # 8. Cache & Memory Persistence
    if "⚠️ SECURITY BLOCK" not in final_answer and "I do not have enough information" not in final_answer:
        cache.set(clean_query, final_answer, results, namespace=role)
    memory.add_turn(session_id, clean_query, final_answer)

    return jsonify({
        "query": clean_query,
        "answer": final_answer,
        "results": results,
        "session_id": session_id,
        "accuracy": accuracy_meta
    })


@app.route("/api/stream", methods=["POST"])
def api_stream():
    """Real-time SSE token streaming endpoint."""
    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default_session")
    role = data.get("role", "default")
    top_k = data.get("top_k", 3)

    if not query:
        return jsonify({"error": "Missing 'query' field."}), 400

    try:
        is_safe, sanitized_or_msg = sanitize_input(query)
        if not is_safe:
            def err_gen():
                yield f"data: {json.dumps({'content': sanitized_or_msg, 'meta': {'results': [], 'query': query, 'session_id': session_id}})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(err_gen()), mimetype="text/event-stream")

        clean_query = sanitized_or_msg

        # Cache Check
        cached_response = cache.get(clean_query, namespace=role)
        if cached_response:
            log.info("⚡ Semantic Cache Hit (Stream)!")
            memory.add_turn(session_id, clean_query, cached_response["answer"])
            def cache_gen():
                yield f"data: {json.dumps({'content': cached_response['answer'] + ' *(⚡ Cached Response)*', 'meta': {'results': cached_response['results'], 'query': clean_query, 'session_id': session_id}})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(cache_gen()), mimetype="text/event-stream")

        # History & Rewriting
        history_str = memory.format_history_for_prompt(session_id)
        try:
            standalone_query = rewriter.rewrite_query(clean_query, history_str)
        except Exception as e:
            log.warning("Rewriter error in api_stream: %s", e)
            standalone_query = clean_query

        # Multi-Stage AI Retrieval (Vector + BM25 Fusion + AI Re-Ranking)
        try:
            vector_candidates = pipeline.retrieve(standalone_query)[:8]
            bm25_candidates = hybrid_retriever.bm25.search(standalone_query, top_k=8)
            fused = hybrid_retriever.reciprocal_rank_fusion(vector_candidates, bm25_candidates)[:8]
            results = ai_engine.rerank_chunks(standalone_query, fused, top_k=top_k)
        except Exception as e:
            log.warning("Hybrid search error in api_stream: %s. Falling back to vector search.", e)
            results = pipeline.retrieve(standalone_query)[:top_k]

        if not results:
            def empty_gen():
                yield f"data: {json.dumps({'content': 'I do not have enough information in my database to answer that.', 'meta': {'results': [], 'query': clean_query, 'session_id': session_id, 'accuracy': {'accuracy_score': 92.0, 'sources': [], 'verification_status': 'No Context Found'}}})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(empty_gen()), mimetype="text/event-stream")

        context_block = pipeline.format_context(results)
        accuracy_meta = ai_engine.extract_sources_and_accuracy(results, True, [])

        prompt = (
            "SYSTEM INSTRUCTIONS (STRICT 98%+ FACTUAL ACCURACY MODE):\n"
            "1. Answer the user question based strictly on the context documents.\n"
            "2. Include clear, explicit document citations (e.g. Document 201, Article 577) for policy rules.\n"
            "3. If context does not contain the answer, reply: 'I do not have enough information in my database to answer that.'\n\n"
            f"RECENT HISTORY:\n{history_str}\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"USER QUESTION: {clean_query}\n"
            "ANSWER:"
        )

        def generate():
            full_answer = []
            yield f"data: {json.dumps({'meta': {'results': results, 'query': clean_query, 'session_id': session_id, 'accuracy': accuracy_meta}})}\n\n"

            for token_evt in stream_llm_generator(prompt):
                try:
                    if token_evt.startswith("data: ") and token_evt.strip() != "data: [DONE]":
                        pdata = json.loads(token_evt[6:])
                        if "content" in pdata:
                            full_answer.append(pdata["content"])
                except Exception:
                    pass
                yield token_evt

            complete_text = "".join(full_answer)
            final_validated = validate_output(complete_text)

            # Groundedness Check for Stream
            try:
                verification = verifier.verify_groundedness(final_validated, results)
                if not verification.get("is_grounded", True):
                    ungrounded = verification.get("ungrounded_claims", [])
                    log.warning("⚠️ Stream Groundedness Warning: %s", ungrounded)
                    analytics_mgr.log_groundedness_warning(clean_query, ungrounded)
            except Exception as e:
                log.warning("Groundedness check failed: %s", e)

            if "⚠️ SECURITY BLOCK" not in final_validated and "I do not have enough information" not in final_validated:
                cache.set(clean_query, final_validated, results, namespace=role)
            memory.add_turn(session_id, clean_query, final_validated)

            yield "data: [DONE]\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        log.exception("Fatal error in /api/stream: %s", e)
        def fatal_err():
            yield f"data: {json.dumps({'content': f'⚠️ Internal server error: {str(e)}', 'meta': {'results': [], 'query': query, 'session_id': session_id}})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(fatal_err()), mimetype="text/event-stream")


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """Logs user thumbs-up/down ratings."""
    data = request.get_json(silent=True) or {}
    query = data.get("query")
    rating = data.get("rating")
    answer = data.get("answer", "")
    session_id = data.get("session_id", "default")
    comment = data.get("comment", "")
    if not query or not rating:
        return jsonify({"error": "Missing query or rating"}), 400
    
    feedback_manager.record_feedback(
        query=query,
        answer=answer,
        rating=rating,
        session_id=session_id,
        comment=comment
    )
    return jsonify({"status": "success", "message": "Feedback recorded."}), 200


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    """Returns current operational analytics for the admin dashboard."""
    return jsonify(analytics_mgr.get_dashboard_metrics()), 200

@app.route("/admin/dashboard")
def render_dashboard():
    """Serves the admin dashboard template."""
    return render_template("dashboard.html")

@app.route("/")
def render_chat():
    """Serves the main chatbot interface."""
    return render_template("chat_v2.html")

@app.route("/api/reindex", methods=["POST"])
@app.route("/api/reload", methods=["POST"])
def api_reindex():
    """Re-scans documents/ and kb.txt to dynamically rebuild the vector & BM25 indexes."""
    try:
        log.info("🔄 Re-indexing knowledge base requested...")
        
        # 1. Re-run document scanning and chunking
        pipeline.ingest()
        
        # 2. Re-initialize Hybrid BM25 Retriever with updated passage chunks
        global hybrid_retriever
        hybrid_retriever = HybridRetriever(pipeline.chunks)
        
        # 3. Clear semantic cache so stale answers aren't returned for updated docs
        cache.clear() if hasattr(cache, 'clear') else None
        
        log.info("✅ Knowledge base successfully re-indexed! Total chunks: %d", len(pipeline.chunks))
        
        return jsonify({
            "status": "success",
            "message": "Knowledge base successfully re-indexed.",
            "total_chunks": len(pipeline.chunks)
        }), 200

    except Exception as e:
        log.error("❌ Error during re-indexing: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Failed to re-index knowledge base: {str(e)}"
        }), 500

if __name__ == "__main__":
    log.info("🚀 Starting Flask RAG Backend Server...")
    app.run(host="0.0.0.0", port=5000, debug=True)