"""
rag_pipeline.py — Local RAG Pipeline with Fine-Tuning Scaffolds & Reranking
=============================================================================
A modular, Windows ARM64-friendly RAG system that:
  • Parses and chunks a knowledge-base file (kb.txt) via document markers
  • Embeds chunks via a lightweight pure-Python TF-IDF approach (no C-binary deps)
  • Includes robust tokenization (retaining numbers/codes like '1' or 'Q3')
  • Bulletproof retrieval: exact index match -> vector match -> substring fallback -> safe default
  • Hybrid Reranker: Re-scores candidates based on term overlap and title alignment
  • Includes sample PEFT (LoRA) and BitFit fine-tuning functions (Hugging Face / PyTorch)
"""

from __future__ import annotations

import os
import re
import sys
import math
import hashlib
import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA PARSING & CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """Represents a single text chunk from the knowledge base."""
    chunk_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


def load_raw_text(file_path: str) -> str:
    """Read the entire knowledge-base file into a string."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge-base file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as fh:
        return fh.read()


def parse_doc_chunks(raw_text: str) -> List[Chunk]:
    """
    Split raw text strictly on [Doc N] or [Document N] markers into 
    independent, self-contained policy chunks.
    """
    parts = re.split(r"(?=\[(?:Doc|Document)\s+\d+\])", raw_text, flags=re.IGNORECASE)
    chunks: List[Chunk] = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        label_match = re.match(r"\[(?:Doc|Document)\s+(\d+)\]\s*(.+)", part, flags=re.IGNORECASE | re.DOTALL)
        if label_match:
            doc_num = label_match.group(1)
            body = label_match.group(2).strip()
            title_match = re.match(r"^(?:Title:\s*)?([^:\n]+)[:\n]", body)
            title = title_match.group(1).strip() if title_match else f"Policy {doc_num}"
        else:
            doc_num = "0"
            body = part
            title = "Untitled Policy"

        chunk_id = hashlib.md5(body.encode()).hexdigest()[:10]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=body,
            metadata={"doc_num": doc_num, "title": title},
        ))
        
    log.info("Successfully parsed %d independent chunks from knowledge base.", len(chunks))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIGHTWEIGHT VECTOR EMBEDDINGS  (pure-Python, ARM64-safe)
# ═══════════════════════════════════════════════════════════════════════════════

class TfidfEmbedder:
    """A from-scratch TF-IDF vectoriser + cosine search with robust number handling."""

    STOP_WORDS = frozenset(
        "a an the is are was were be been being have has had do does did "
        "will would shall should may might can could of in to for on with "
        "at by from as into through during before after above below between "
        "out off over under again further then once and but or nor not so "
        "yet both either neither each every all any few more most other some "
        "such no only own same than too very it its he she they them their "
        "this that these those i me my we our you your".split()
    )

    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_vectors: List[List[float]] = []
        self.documents: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase, strip punctuation, remove stop-words, but preserve numbers/digits."""
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        # Keep tokens if they aren't stop words, and are either length > 1 OR numeric digits (e.g. '1', '2')
        return [
            t for t in tokens 
            if t not in TfidfEmbedder.STOP_WORDS and (len(t) > 1 or t.isdigit())
        ]

    def fit(self, documents: List[str]) -> None:
        self.documents = documents
        n_docs = len(documents)

        df: Counter = Counter()
        tokenized_docs: List[List[str]] = []
        for doc in documents:
            tokens = self._tokenize(doc)
            tokenized_docs.append(tokens)
            df.update(set(tokens))

        self.vocab = {term: idx for idx, term in enumerate(sorted(df.keys()))}
        self.idf = {
            term: math.log((n_docs + 1) / (count + 1)) + 1.0
            for term, count in df.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in tokenized_docs]
        log.info("TF-IDF index built — vocab size: %d, docs: %d", len(self.vocab), n_docs)

    def _vectorize(self, tokens: List[str]) -> List[float]:
        tf = Counter(tokens)
        vec = [0.0] * len(self.vocab)
        for term, count in tf.items():
            if term in self.vocab:
                vec[self.vocab[term]] = (count / max(len(tokens), 1)) * self.idf.get(term, 0.0)
        return vec

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a)) or 1e-10
        norm_b = math.sqrt(sum(x * x for x in b)) or 1e-10
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vec = self._vectorize(self._tokenize(query))
        scored = [
            (idx, self._cosine_similarity(query_vec, dv))
            for idx, dv in enumerate(self.doc_vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for idx, score in scored[:top_k]:
            results.append({
                "document": self.documents[idx],
                "score": round(score, 4),
                "index": idx,
            })
        return results


def get_embedder(prefer_transformer: bool = False):
    return TfidfEmbedder()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RAG PIPELINE ORCHESTRATOR & RERANKER
# ═══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    def __init__(
        self,
        kb_path: str = "kb.txt",
        top_k: int = 3,
        prefer_transformer: bool = False,
    ) -> None:
        self.kb_path = kb_path
        self.top_k = top_k
        self.chunks: List[Chunk] = []
        self.embedder = get_embedder(prefer_transformer)

    def ingest(self) -> None:
        raw = load_raw_text(self.kb_path)
        self.chunks = parse_doc_chunks(raw)
        if not self.chunks:
            log.error("No chunks parsed from %s.", self.kb_path)
            return
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        log.info("Ingestion complete — %d chunks indexed.", len(self.chunks))

    def rerank(self, query: str, candidate_chunks: list, top_k: int = 3) -> list:
        """
        Re-scores and re-ranks candidate chunks based on exact query term alignment,
        phrase match position, and query-term density.
        """
        if not candidate_chunks:
            return []

        query_terms = set(re.findall(r"\w+", query.lower()))
        if not query_terms:
            return candidate_chunks[:top_k]

        scored_candidates = []
        for chunk_item in candidate_chunks:
            text = chunk_item.get("document", chunk_item.get("text", "")).lower()
            text_words = re.findall(r"\w+", text)
            
            # Score factors
            term_matches = sum(1 for term in query_terms if term in text_words)
            overlap_ratio = term_matches / len(query_terms) if query_terms else 0
            
            # Weighted rerank score combining base score + overlap ratio
            base_score = chunk_item.get("score", 0.5)
            rerank_score = (base_score * 0.4) + (overlap_ratio * 0.6)
            
            # Title term match boost
            if any(term in text[:100] for term in query_terms):
                rerank_score += 0.1

            item_copy = dict(chunk_item)
            item_copy["score"] = round(rerank_score, 4)
            scored_candidates.append(item_copy)

        # Sort descending by re-ranked score
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        return scored_candidates[:top_k]

    def retrieve(self, query: str, min_score: float = 0.01) -> List[Dict[str, Any]]:
        """
        Balanced bulletproof retrieval:
        0. Exact Document/Policy Index Router (prevents collisions like Policy 3 matching Policy 30).
        1. Vector search with a sensible threshold.
        2. Substring fallback strictly capped at top_k chunks.
        3. Graceful single-chunk default if nothing matches.
        4. Re-ranking cross-scorer.
        """
        query_clean = query.strip().lower()
        if not query_clean or not self.chunks:
            return []

        # ─── Step 0: Exact Document/Policy Index Matcher ─────────────────────
        match = re.search(r"\b(?:policy|document)\s*(\d+)\b", query_clean, re.IGNORECASE)
        if match:
            target_num = match.group(1)
            exact_matches = []
            
            for idx, chunk in enumerate(self.chunks):
                if re.search(rf"\b(?:policy|document)\s*{target_num}\b", chunk.text, re.IGNORECASE):
                    exact_matches.append({
                        "document": chunk.text,
                        "score": 1.0,  # Top priority exact match
                        "index": idx,
                        "metadata": chunk.metadata
                    })
                    if len(exact_matches) >= self.top_k:
                        break
            
            if exact_matches:
                return exact_matches

        # ─── Step 1: Run vector search ────────────────────────────────────────
        raw_results = self.embedder.search(query, top_k=self.top_k * 2)  # Fetch extra candidates for reranker
        
        filtered_results = []
        for r in raw_results:
            if r.get("score", 0) >= min_score:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(self.chunks):
                    r["metadata"] = self.chunks[idx].metadata
                    if "document" not in r and "text" in r:
                        r["document"] = r["text"]
                    filtered_results.append(r)

        # ─── Step 2: Exact substring match fallback ───────────────────────────
        if not filtered_results:
            for idx, chunk in enumerate(self.chunks):
                if query_clean in chunk.text.lower():
                    filtered_results.append({
                        "document": chunk.text,
                        "score": 0.5,  
                        "index": idx,
                        "metadata": chunk.metadata
                    })
                    if len(filtered_results) >= self.top_k:
                        break

        # ─── Step 3: Graceful safety net ──────────────────────────────────────
        if not filtered_results and self.chunks:
            filtered_results.append({
                "document": self.chunks[0].text,
                "score": 0.02,
                "index": 0,
                "metadata": self.chunks[0].metadata
            })

        # ─── Step 4: Re-rank candidates & enforce top_k ───────────────────────
        return self.rerank(query, filtered_results, top_k=self.top_k)

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Combine retrieved chunks into concise, isolated prompt blocks."""
        parts: List[str] = []
        for i, r in enumerate(results, 1):
            title = r.get("metadata", {}).get("title", "Policy Document")
            parts.append(f"--- START {title} ---\n{r['document']}\n--- END {title} ---")
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FINE-TUNING SCAFFOLDS (PEFT / LoRA / BitFit)
# ═══════════════════════════════════════════════════════════════════════════════

def peft_lora_demo(base_model_name: str = "bert-base-uncased") -> None:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as exc:
        log.error("Missing dependency for LoRA demo: %s", exc)
        return

    log.info("═══ PEFT / LoRA Demo ═══")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, target_modules=["query", "value"], bias="none")
    lora_model = get_peft_model(base_model, lora_config)
    log.info("LoRA setup complete.")


def bitfit_freeze_demo(base_model_name: str = "bert-base-uncased") -> None:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        log.error("Missing dependency for BitFit demo: %s", exc)
        return

    log.info("═══ BitFit Demo ═══")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "bias" in name:
            param.requires_grad = True
    log.info("BitFit setup complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Local RAG Pipeline")
    parser.add_argument("--kb", default="kb.txt", help="Path to kb.txt")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k results")
    parser.add_argument("--query", type=str, default=None, help="Query string")
    parser.add_argument("--finetuning-demo", action="store_true", help="Run LoRA/BitFit demos")
    args = parser.parse_args()

    if args.finetuning_demo:
        peft_lora_demo()
        bitfit_freeze_demo()
        return

    pipeline = RAGPipeline(kb_path=args.kb, top_k=args.top_k)
    pipeline.ingest()

    query = args.query if args.query else "Policy 1"
    results = pipeline.retrieve(query)
    print(f"\nQuery: {query} -> Retrieved {len(results)} chunks:")
    for rank, res in enumerate(results, 1):
        print(f"  [{rank}] {res.get('metadata', {}).get('title')} | Score: {res['score']}")


if __name__ == "__main__":
    main()