"""
semantic_cache.py — Thread-Safe Atomic Semantic Cache with Invalidation & Multi-Tenancy
=======================================================================================
Stores and retrieves semantically similar past queries using TF-IDF / Cosine Similarity with:
  • Thread-safe atomic file writes (prevents JSON corruption on concurrent requests)
  • Namespace isolation for multi-tenancy
  • Instant invalidation capabilities on KB reload
"""

import json
import os
import re
import math
import time
import logging
import threading
from collections import Counter

log = logging.getLogger(__name__)

CACHE_FILE = "semantic_cache.json"


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 or t.isdigit()]


def _compute_cosine_similarities(query: str, cached_queries: list[str]) -> list[float]:
    docs = cached_queries + [query]
    tokenized = [_tokenize(d) for d in docs]
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
    n = len(docs)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def vectorize(tokens: list[str]) -> list[float]:
        tf = Counter(tokens)
        vec = [0.0] * len(vocab)
        for t, cnt in tf.items():
            if t in vocab:
                vec[vocab[t]] = (cnt / max(len(tokens), 1)) * idf[t]
        return vec

    vectors = [vectorize(tok) for tok in tokenized]
    q_vec = vectors[-1]

    def cos_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-10
        nb = math.sqrt(sum(y * y for y in b)) or 1e-10
        return dot / (na * nb)

    return [cos_sim(q_vec, v) for v in vectors[:-1]]


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.82, cache_file: str = CACHE_FILE):
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.entries = []  # List of {"namespace": str, "query": str, "answer": str, "results": list, "timestamp": float}
        self._lock = threading.Lock()
        self.load_from_disk()

    def get(self, query: str, namespace: str = "default") -> dict | None:
        """Find a cached entry with similarity exceeding the threshold within the same namespace."""
        with self._lock:
            # Filter entries matching current namespace
            valid_entries = [e for e in self.entries if e.get("namespace", "default") == namespace]
            if not valid_entries:
                return None

            try:
                cached_queries = [e["query"] for e in valid_entries]
                similarities = _compute_cosine_similarities(query, cached_queries)
                if not similarities:
                    return None

                best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
                best_score = float(similarities[best_idx])

                if best_score >= self.similarity_threshold:
                    match = valid_entries[best_idx]
                    return {
                        "query": match["query"],
                        "answer": match["answer"],
                        "results": match["results"],
                        "similarity_score": round(best_score, 3)
                    }
            except Exception as e:
                log.error("Error computing semantic cache match: %s", e)

            return None

    def set(self, query: str, answer: str, results: list, namespace: str = "default"):
        """Store a new query answer pair in the cache and save atomically."""
        with self._lock:
            entry = {
                "namespace": namespace,
                "query": query,
                "answer": answer,
                "results": results,
                "timestamp": time.time()
            }
            self.entries.append(entry)
            self._save_to_disk_atomic()

    def clear(self):
        """Wipe all cached entries (called automatically on KB reload)."""
        with self._lock:
            self.entries = []
            self._save_to_disk_atomic()
            log.info("⚡ Semantic Cache fully cleared.")

    def _save_to_disk_atomic(self):
        """Thread-safe atomic write to disk to prevent file corruption."""
        temp_file = f"{self.cache_file}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, self.cache_file)
        except Exception as e:
            log.error("Failed atomic write to semantic cache file: %s", e)
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def load_from_disk(self):
        """Load cache entries safely from disk."""
        with self._lock:
            if os.path.exists(self.cache_file):
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        self.entries = json.load(f)
                    log.info("Loaded %d semantic cache entries.", len(self.entries))
                except Exception as e:
                    log.error("Failed to load semantic cache file: %s", e)
                    self.entries = []