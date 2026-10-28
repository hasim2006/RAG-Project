"""
memory_manager.py — Thread-Safe Multi-Turn Memory Manager with LRU & TTL Eviction
===================================================================================
Manages multi-turn conversation context per session with:
  • Max turn limits per session
  • Automatic Time-To-Live (TTL) session cleanup to prevent RAM memory leaks
  • Thread-safe atomic access
"""

import time
import threading
import logging
from collections import OrderedDict

log = logging.getLogger(__name__)


class MemoryManager:
    def __init__(self, max_turns: int = 5, max_sessions: int = 1000, ttl_seconds: int = 86400):
        """
        :param max_turns: Max conversation turns to keep per session.
        :param max_sessions: Maximum concurrent sessions allowed in RAM (LRU eviction).
        :param ttl_seconds: Inactivity timeout before a session is automatically purged (default: 24 hrs).
        """
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions = OrderedDict()  # Stores session_id -> {"turns": [...], "last_accessed": timestamp}
        self._lock = threading.Lock()

    @property
    def sessions(self) -> OrderedDict:
        """Expose sessions dictionary for monitoring/analytics."""
        return self._sessions

    def _cleanup_expired_sessions(self):
        """Purge sessions that have exceeded the TTL limit."""
        now = time.time()
        expired_keys = [
            sid for sid, data in self._sessions.items()
            if now - data["last_accessed"] > self.ttl_seconds
        ]
        for sid in expired_keys:
            del self._sessions[sid]
            log.info("Purged expired session memory: '%s'", sid)

    def add_turn(self, session_id: str, user_query: str, ai_response: str):
        """Record a single query/response turn in the session history."""
        with self._lock:
            self._cleanup_expired_sessions()

            now = time.time()
            if session_id in self._sessions:
                # Move to end (most recently used)
                self._sessions.move_to_end(session_id)
                session_data = self._sessions[session_id]
            else:
                # Evict oldest session if cap is reached
                if len(self._sessions) >= self.max_sessions:
                    oldest_sid, _ = self._sessions.popitem(last=False)
                    log.warning("Evicted oldest session from memory cap: '%s'", oldest_sid)
                
                session_data = {"turns": [], "last_accessed": now}
                self._sessions[session_id] = session_data

            session_data["last_accessed"] = now
            session_data["turns"].append({
                "user": user_query,
                "ai": ai_response,
                "timestamp": now
            })

            # Trim turns to max_turns window
            if len(session_data["turns"]) > self.max_turns:
                session_data["turns"] = session_data["turns"][-self.max_turns:]

    def get_history(self, session_id: str) -> list[dict]:
        """Retrieve historical turns for a specific session."""
        with self._lock:
            self._cleanup_expired_sessions()
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                self._sessions[session_id]["last_accessed"] = time.time()
                return list(self._sessions[session_id]["turns"])
            return []

    def format_history_for_prompt(self, session_id: str) -> str:
        """Format history turns into a clean context string for LLM prompts."""
        turns = self.get_history(session_id)
        if not turns:
            return "None (This is the start of the conversation)."

        formatted = []
        for i, turn in enumerate(turns, 1):
            formatted.append(f"Turn {i}:\n  User: {turn['user']}\n  Assistant: {turn['ai']}")
        return "\n\n".join(formatted)

    def clear_session(self, session_id: str):
        """Clear memory for a specific session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                log.info("Cleared memory for session: '%s'", session_id)