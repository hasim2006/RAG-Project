# Enterprise RAG Chatbot & Policy Retrieval System

A modular Retrieval-Augmented Generation (RAG) backend and interactive chat interface integrating an **AI Voice Agent** (Speech-to-Text & Text-to-Speech), Guardrails, Semantic Caching, Multi-Turn Memory, Hybrid Search (BM25 + TF-IDF Vector Retrieval + Reciprocal Rank Fusion), Groundedness Verification, Real-Time SSE Token Streaming, and an Operational Analytics Dashboard.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Free Groq API Key (from [console.groq.com](https://console.groq.com/))

### 2. Setup Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Copy `.env.example` to `.env` and insert your Groq API key:
```env
GROQ_API_KEY="gsk_your_actual_key_here"
```

### 5. Launch the Application
```bash
python app.py
```
Open your browser and navigate to:
- **Chatbot Interface:** [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- **System Health & Analytics Dashboard:** [http://127.0.0.1:5000/admin/dashboard](http://127.0.0.1:5000/admin/dashboard)

---

## 📁 Project Architecture & Files

- `app.py`: Flask application server with REST & SSE endpoints (`/api/query`, `/api/stream`, `/api/feedback`, `/api/analytics`, `/api/reindex`).
- `rag_pipeline.py`: Document ingestion, multi-format file extraction (.txt, .md, .pdf, .docx), and bulletproof multi-stage retriever.
- `hybrid_search.py`: Hybrid search engine combining Robertson-Spärck Jones BM25 keyword matching with TF-IDF vector similarity via Reciprocal Rank Fusion (RRF).
- `semantic_cache.py`: Thread-safe atomic semantic cache using cosine similarity with namespace isolation.
- `memory_manager.py`: Thread-safe multi-turn conversation memory with LRU session management and TTL eviction.
- `feedback_manager.py`: User satisfaction rating engine (thumbs up/down) with thread-safe atomic disk persistence and fine-tuning queue.
- `groundedness_verifier.py`: LLM-based hallucination auditor checking if answers are faithful to retrieved context.
- `query_rewriter.py`: Multi-turn conversational query reformulation for standalone search retrieval.
- `analytics_manager.py`: Aggregator for system health metrics (cache hits, active sessions, groundedness alerts, satisfaction rate).
- `run_evals.py`: Automated retrieval precision/recall and faithfulness benchmark suite.
- `templates/`: Modern, responsive web user interfaces (`chat_v2.html` and `dashboard.html`).
- `documents/`: Knowledge base documents across PDF, DOCX, MD, and TXT formats.
- `kb.txt`: Standard enterprise policy knowledge base file.
- `generate_sample_docs.py`: Utility script to regenerate multi-format corporate policy files.
