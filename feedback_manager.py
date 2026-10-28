"""
feedback_manager.py — Thread-Safe Asynchronous Feedback & Continuous Learning Loop
=====================================================================================
Collects user feedback (thumbs up/down) on AI answers and:
  1. Stores logs persistently with thread-safe atomic disk writes
  2. Queues negative/disliked responses for manual review or model fine-tuning
  3. Returns real-time satisfaction analytics
"""

import json
import os
import time
import logging
import threading

log = logging.getLogger(__name__)

FEEDBACK_FILE = "feedback_store.json"
FINETUNE_QUEUE_FILE = "dataset_for_finetuning.jsonl"


class FeedbackManager:
    def __init__(self):
        self.feedback_logs = []
        self._lock = threading.Lock()
        self.load_from_disk()

    def record_feedback(self, query: str, answer: str, rating: str, session_id: str = "default", comment: str = ""):
        """Record user feedback with thread-safe disk writing."""
        entry = {
            "timestamp": time.time(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "rating": rating,
            "comment": comment
        }
        
        with self._lock:
            self.feedback_logs.append(entry)
            self._save_to_disk_atomic()

        # Queue negative feedback for fine-tuning
        if str(rating).lower() in ["negative", "down", "dislike"]:
            self._queue_for_finetuning(entry)

        log.info("Recorded %s feedback for query: '%s'", str(rating).upper(), query)

    def _queue_for_finetuning(self, entry: dict):
        """Append negative feedback instances safely to the fine-tuning queue."""
        try:
            with self._lock:
                with open(FINETUNE_QUEUE_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.info("Queued query to '%s' for model retraining review.", FINETUNE_QUEUE_FILE)
        except Exception as e:
            log.error("Failed to append to fine-tuning queue: %s", e)

    def add_feedback(self, query: str, rating: str, answer: str = "", session_id: str = "default", comment: str = ""):
        """Alias for record_feedback for backward compatibility."""
        return self.record_feedback(query=query, answer=answer, rating=rating, session_id=session_id, comment=comment)

    def get_analytics(self) -> dict:
        """Return analytics summary of user satisfaction."""
        with self._lock:
            if not self.feedback_logs:
                return {
                    "total_feedback": 0,
                    "positive": 0,
                    "negative": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "satisfaction_rate": "0%"
                }

            total = len(self.feedback_logs)
            pos = sum(1 for item in self.feedback_logs if str(item.get("rating", "")).lower() in ["positive", "up", "like"])
            neg = total - pos
            rate = round((pos / total) * 100, 1)

            return {
                "total_feedback": total,
                "positive": pos,
                "negative": neg,
                "positive_count": pos,
                "negative_count": neg,
                "satisfaction_rate": f"{rate}%"
            }

    def _save_to_disk_atomic(self):
        """Perform thread-safe atomic file writing."""
        temp_file = f"{FEEDBACK_FILE}.tmp"
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.feedback_logs, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, FEEDBACK_FILE)
        except Exception as e:
            log.error("Failed to save feedback to disk: %s", e)
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def load_from_disk(self):
        """Load past feedback logs safely from disk."""
        with self._lock:
            if os.path.exists(FEEDBACK_FILE):
                try:
                    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                        self.feedback_logs = json.load(f)
                    log.info("Loaded %d feedback logs from disk.", len(self.feedback_logs))
                except Exception as e:
                    log.error("Failed to load feedback logs: %s", e)
                    self.feedback_logs = []