"""
rag_pipeline.py — Multi-Format & Marker-Based RAG Pipeline
===========================================================
Combines:
  • Document marker parsing ([Doc N] / [Document N]) for structured policies (kb.txt)
  • Directory scanning for multi-format documents (.pdf, .docx, .md, .txt)
  • Pure-Python ARM64-safe TfidfEmbedder with number/digit preservation
  • Multi-stage bulletproof retrieval (Exact Policy Index -> Vector Search -> Substring -> Fallback)
  • Fine-tuning scaffolds (LoRA / BitFit)
"""

from __future__ import annotations

import os
import re
import sys
import glob
import math
import hashlib
import argparse
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Multi-format Document Extractors
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

# ─── Logging Setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — MULTI-FORMAT FILE EXTRACTORS & CHUNKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    """Represents a single text chunk from the knowledge base."""
    chunk_id: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.chunk_id)

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return self.chunk_id == other.chunk_id
        return False

    def __str__(self):
        return self.text


def extract_text_from_file(file_path: str) -> str:
    """Extracts raw plain text from PDF, DOCX, MD, and TXT files."""
    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    try:
        if ext == ".pdf":
            if PdfReader is None:
                log.error("pypdf is not installed. Skipping PDF: %s", file_path)
                return ""
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        elif ext == ".docx":
            if docx is None:
                log.error("python-docx is not installed. Skipping DOCX: %s", file_path)
                return ""
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs if p.text])

        elif ext in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

    except Exception as e:
        log.error("Error reading %s: %s", file_path, e)

    return text.strip()


def parse_doc_chunks(raw_text: str, default_source: str = "kb.txt") -> List[Chunk]:
    """
    Splits raw text on [Doc N] or [Document N] markers if present;
    otherwise falls back to logical paragraph/block chunks.
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
            full_text = f"[Document {doc_num}] {body}"
        else:
            doc_num = "0"
            body = part
            title = os.path.basename(default_source)
            full_text = body

        chunk_id = hashlib.md5(full_text.encode()).hexdigest()[:10]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=full_text,
            metadata={"doc_num": doc_num, "title": title, "source": default_source},
        ))

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LIGHTWEIGHT VECTOR EMBEDDINGS (pure-Python, ARM64-safe)
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
# SECTION 3 — RAG PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    def __init__(
        self,
        kb_path: str = "kb.txt",
        docs_dir: str = "documents",
        top_k: int = 3,
        prefer_transformer: bool = False,
    ) -> None:
        self.kb_path = kb_path
        self.docs_dir = docs_dir
        self.top_k = top_k
        self.chunks: List[Chunk] = []
        self.embedder = get_embedder(prefer_transformer)

    def ingest(self) -> None:
        """Loads, parses, and indexes documents from kb.txt and documents/ directory."""
        self.chunks = []

        # 1. Parse kb.txt if present
        if os.path.exists(self.kb_path):
            raw_text = extract_text_from_file(self.kb_path)
            if raw_text:
                parsed = parse_doc_chunks(raw_text, default_source=self.kb_path)
                self.chunks.extend(parsed)

        # 2. Parse all files in documents/ directory
        if os.path.exists(self.docs_dir):
            supported_extensions = ["*.txt", "*.md", "*.pdf", "*.docx"]
            for ext in supported_extensions:
                pattern = os.path.join(self.docs_dir, "**", ext)
                for file_path in glob.glob(pattern, recursive=True):
                    file_text = extract_text_from_file(file_path)
                    if file_text:
                        parsed = parse_doc_chunks(file_text, default_source=file_path)
                        self.chunks.extend(parsed)

        if not self.chunks:
            log.error("No valid chunks ingested across knowledge bases.")
            return

        log.info("Successfully parsed %d independent chunks from knowledge base.", len(self.chunks))
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        log.info("Ingestion complete — %d chunks indexed.", len(self.chunks))

    def retrieve(self, query: str, min_score: float = 0.01) -> List[Dict[str, Any]]:
        """
        Multi-stage Bulletproof Retrieval:
        0. Exact Document/Policy Index Router (e.g. Policy 1 vs Policy 10).
        1. Vector similarity search.
        2. Exact substring match fallback.
        3. Single-chunk safe default fallback.
        """
        query_clean = re.sub(r"[\u202f\xa0\s]+", " ", query.strip().lower())
        if not query_clean or not self.chunks:
            return []

        # ─── Step 0: Exact Document/Policy Index Matcher ─────────────────────
        match = re.search(r"\b(?:policy|document|doc)\s*(\d+)\b", query_clean, re.IGNORECASE)
        if match:
            target_num = match.group(1)
            exact_matches = []

            for idx, chunk in enumerate(self.chunks):
                doc_num = str(chunk.metadata.get("doc_num", ""))
                if doc_num == target_num or re.search(rf"\b(?:policy|document|doc)\s*{target_num}\b", chunk.text, re.IGNORECASE):
                    exact_matches.append({
                        "document": chunk.text,
                        "score": 1.0,
                        "index": idx,
                        "metadata": chunk.metadata
                    })
                    if len(exact_matches) >= self.top_k:
                        break

            if exact_matches:
                return exact_matches

        # ─── Step 1: Vector Search ───────────────────────────────────────────
        raw_results = self.embedder.search(query, top_k=self.top_k)

        filtered_results = []
        for r in raw_results:
            if r.get("score", 0) >= min_score:
                idx = r.get("index")
                if idx is not None and 0 <= idx < len(self.chunks):
                    r["metadata"] = self.chunks[idx].metadata
                    if "document" not in r and "text" in r:
                        r["document"] = r["text"]
                    filtered_results.append(r)

        # ─── Step 2: Exact Substring Fallback ────────────────────────────────
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

        # ─── Step 3: Graceful Safety Net ─────────────────────────────────────
        if not filtered_results and self.chunks:
            filtered_results.append({
                "document": self.chunks[0].text,
                "score": 0.02,
                "index": 0,
                "metadata": self.chunks[0].metadata
            })

        return filtered_results[:self.top_k]

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Combines retrieved chunks into concise, isolated prompt context blocks."""
        parts: List[str] = []
        for r in results:
            meta = (r.get("metadata") or {}) if isinstance(r, dict) else getattr(r, "metadata", {}) or {}
            title = meta.get("title", "Policy Document")
            doc_num = str(meta.get("doc_num", ""))
            doc_label = f"Document {doc_num}: {title}" if doc_num and doc_num != "0" else title
            doc_text = r.get("document", "") if isinstance(r, dict) else (getattr(r, "text", "") or str(r))
            parts.append(f"--- START {doc_label} ---\n{doc_text}\n--- END {doc_label} ---")
        return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FINE-TUNING SCAFFOLDS (PEFT / LoRA / BitFit)
# ═══════════════════════════════════════════════════════════════════════════════

def peft_lora_demo(base_model_name: str = "bert-base-uncased") -> None:
    try:
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

    parser = argparse.ArgumentParser(description="Local Multi-Format RAG Pipeline")
    parser.add_argument("--kb", default="kb.txt", help="Path to kb.txt")
    parser.add_argument("--docs-dir", default="documents", help="Directory containing PDF/DOCX/MD files")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k results")
    parser.add_argument("--query", type=str, default=None, help="Query string")
    parser.add_argument("--finetuning-demo", action="store_true", help="Run LoRA/BitFit demos")
    args = parser.parse_args()

    if args.finetuning_demo:
        peft_lora_demo()
        bitfit_freeze_demo()
        return

    pipeline = RAGPipeline(kb_path=args.kb, docs_dir=args.docs_dir, top_k=args.top_k)
    pipeline.ingest()

    query = args.query if args.query else "Policy 1"
    results = pipeline.retrieve(query)
    print(f"\nQuery: {query} -> Retrieved {len(results)} chunks:")
    for rank, res in enumerate(results, 1):
        print(f"  [{rank}] {res.get('metadata', {}).get('title')} | Score: {res['score']}")


if __name__ == "__main__":
    main()