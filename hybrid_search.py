"""
hybrid_search.py — Hybrid Search Engine (BM25 + TF-IDF Vector Search + RRF)
=============================================================================
Combines sparse keyword scoring (BM25) with dense/TF-IDF similarity and 
fuses them using Reciprocal Rank Fusion (RRF) for optimal context retrieval.
"""

import math
import logging
from collections import defaultdict, Counter
log = logging.getLogger(__name__)


class BM25Retriever:
    """Lightweight BM25 Implementation for Keyword Matching."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.doc_count = 0
        self.idf = {}
        self.doc_freqs = []
        self.docs = []

    def fit(self, docs: list):
        # Extract text if docs contains Chunk objects instead of strings
        string_docs = [
            doc.text if hasattr(doc, "text") 
            else getattr(doc, "document", str(doc)) 
            for doc in docs
        ]
        self.docs = string_docs
        self.doc_count = len(string_docs)
        if self.doc_count == 0:
            return

        words_per_doc = [doc.lower().split() for doc in string_docs]
        self.doc_len = [len(words) for words in words_per_doc]
        self.avg_doc_len = sum(self.doc_len) / self.doc_count if self.doc_count > 0 else 1.0

        # Compute term frequencies per document
        self.doc_freqs = [Counter(words) for words in words_per_doc]

        # Compute document frequencies across corpus
        df = defaultdict(int)
        for freq in self.doc_freqs:
            for word in freq:
                df[word] += 1

        # Compute Robertson-Spärck Jones IDF
        self.idf = {
            word: math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1.0)
            for word, count in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q_words = query.lower().split()
        scores = []

        for idx, freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_l = self.doc_len[idx]
            for w in q_words:
                if w in freq:
                    w_freq = freq[w]
                    num = self.idf.get(w, 0) * w_freq * (self.k1 + 1)
                    den = w_freq + self.k1 * (1 - self.b + self.b * (doc_l / self.avg_doc_len))
                    score += num / den
            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class HybridRetriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.bm25 = BM25Retriever()
        self.bm25.fit(chunks)

    @staticmethod
    def _extract_doc_info(item) -> tuple[str, dict]:
        """Safely extracts plain text and metadata from Chunk dataclass, dict, or string."""
        if hasattr(item, "text"):
            return item.text, getattr(item, "metadata", {}) or {}
        if isinstance(item, dict):
            text = item.get("document") or item.get("text") or ""
            meta = item.get("metadata") or {}
            return text, meta
        return str(item), {}

    def reciprocal_rank_fusion(self, vector_results: list[dict], bm25_scores: list[tuple[int, float]], k: int = 60) -> list[dict]:
        """Fuses sparse and dense rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = defaultdict(float)
        doc_metadata_map = {}

        # 1. Score vector results
        for rank, res in enumerate(vector_results):
            doc_text, meta = self._extract_doc_info(res)
            if doc_text:
                score_boost = 3.0 if res.get("score", 0) >= 1.0 else 1.0
                rrf_scores[doc_text] += (1.0 / (k + rank + 1)) * score_boost
                if meta and doc_text not in doc_metadata_map:
                    doc_metadata_map[doc_text] = meta

        # 2. Score BM25 results
        for rank, (doc_idx, score) in enumerate(bm25_scores):
            if 0 <= doc_idx < len(self.chunks):
                raw_chunk = self.chunks[doc_idx]
                doc_text, meta = self._extract_doc_info(raw_chunk)
                if doc_text:
                    rrf_scores[doc_text] += 1.0 / (k + rank + 1)
                    if meta and doc_text not in doc_metadata_map:
                        doc_metadata_map[doc_text] = meta

        # 3. Sort and construct fused output
        fused = []
        for doc_text, fused_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            entry = {
                "document": doc_text,
                "score": round(fused_score, 4)
            }
            if doc_text in doc_metadata_map:
                entry["metadata"] = doc_metadata_map[doc_text]
            fused.append(entry)

        return fused