"""
analytics_manager.py — Analytics & Health Monitoring Engine
==========================================================
Aggregates metrics for Groundedness Warnings, Cache Hits, and User Feedback.
"""

import logging

log = logging.getLogger(__name__)


class AnalyticsManager:
    def __init__(self, cache, memory, feedback_mgr):
        self.cache = cache
        self.memory = memory
        self.feedback_mgr = feedback_mgr
        self.groundedness_warnings = []

    def log_groundedness_warning(self, query: str, ungrounded_claims: list):
        """Logs a hallucination/groundedness flag for dashboard review."""
        self.groundedness_warnings.append({
            "query": query,
            "claims": ungrounded_claims
        })
        log.warning("Logged Groundedness Warning for query: '%s'", query)

    def get_dashboard_metrics(self) -> dict:
        """Returns a consolidated summary of system health."""
        feedback_summary = self.feedback_mgr.get_analytics() if hasattr(self.feedback_mgr, 'get_analytics') else {}
        
        return {
            "cache": {
                "total_cached_entries": len(getattr(self.cache, 'entries', {})),
                "similarity_threshold": getattr(self.cache, 'similarity_threshold', 0.85)
            },
            "memory": {
                "active_sessions": len(getattr(self.memory, 'sessions', getattr(self.memory, '_sessions', {})))
            },
            "feedback": feedback_summary,
            "groundedness": {
                "total_warnings": len(self.groundedness_warnings),
                "recent_warnings": self.groundedness_warnings[-5:]
            }
        }