"""Intent router for GraphRAG.

This decides whether a query should go to graph, timeline, vector, or
verification paths. The router is intentionally lightweight so the actual
retrieval code stays reusable and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import re

from .utils import score_query_intent


@dataclass
class RouteDecision:
    intent: str
    rationale: str
    use_vector: bool
    use_graph: bool
    use_timeline: bool
    use_verification: bool
    confidence: float = 0.0


class IntentRouter:
    def classify(self, query: str) -> RouteDecision:
        scores = score_query_intent(query)
        q = query.lower()

        if scores["verify"] >= 2:
            return RouteDecision(
                intent="verification",
                rationale="The query asks for cross-checking or comparison, so use graph, vector, and timeline evidence together.",
                use_vector=True,
                use_graph=True,
                use_timeline=True,
                use_verification=True,
                confidence=0.92,
            )

        if scores["graph"] >= max(scores["timeline"], scores["vector"]):
            return RouteDecision(
                intent="graph",
                rationale="The query is relationship-centric, so graph traversal should be the primary path.",
                use_vector=False,
                use_graph=True,
                use_timeline=False,
                use_verification=False,
                confidence=0.88,
            )

        if scores["timeline"] > 0 and any(token in q for token in ["year", "when", "before", "after", "timeline", "chronology", "during"]):
            return RouteDecision(
                intent="timeline",
                rationale="The query is temporal, so the timeline store is the most direct path.",
                use_vector=False,
                use_graph=False,
                use_timeline=True,
                use_verification=False,
                confidence=0.84,
            )

        return RouteDecision(
            intent="vector",
            rationale="The query looks like a fact lookup, so shared hybrid retrieval is the lightest useful path.",
            use_vector=True,
            use_graph=False,
            use_timeline=False,
            use_verification=False,
            confidence=0.78,
        )


def infer_entities_from_query(query: str) -> List[str]:
    # Common stop words and question words to exclude
    stop_words = {
        "who", "what", "when", "where", "why", "how", "which", "with", "without",
        "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "by", "at",
        "was", "were", "is", "are", "did", "do", "does", "has", "have", "been",
        "from", "that", "this", "these", "those", "it", "they", "we", "you", "he", "she"
    }
    candidates = re.findall(r"\b[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,4}\b", query)
    out = []
    seen = set()
    for c in candidates:
        n = c.strip()
        lower_n = n.lower()
        if lower_n in stop_words:
            continue
        # Also skip if it's a single letter (like "I" but unlikely)
        if len(n) <= 1:
            continue
        key = lower_n
        if key not in seen:
            seen.add(key)
            out.append(n)
    return out
