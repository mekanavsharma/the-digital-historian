"""Shared helpers for phase 2 GraphRAG."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Optional, Tuple

import re

PRONOUNS = {
    "he", "she", "they", "his", "her", "their", "it", "its",
    "him", "this", "that", "these", "those", "the same"
}

RELATION_HINTS = {
    "fought": ["fought", "battle", "war", "against", "defeated", "defeat", "clash"],
    "allied_with": ["allied", "alliance", "ally", "supported", "coalition"],
    "succeeded_by": ["succeeded by", "followed by", "came after", "replaced by"],
    "ruled": ["ruled", "reigned", "governed", "administered", "emperor", "king"],
    "wrote_about": ["wrote", "described", "according to", "argues", "claims", "states"],
    "contemporary_of": ["contemporary", "same period", "around the same time"],
}

HISTORIAN_HINTS = {
    "majumdar", "sarkar", "smith", "tripathi", "raychaudhuri", "thapar",
    "chandra", "habib", "rizvi", "nirad", "banerjee", "panikkar", "khan"
}

DYNASTY_HINTS = {
    "mughal", "maurya", "mauryan", "gupta", "chola", "chalukya",
    "rajas", "delhi sultanate", "khalji", "tughlaq", "lodi", "sur", "maratha",
    "vijayanagara", "sikh", "pala", "rashtrakuta", "sen", "kakatiya",
}

EVENT_HINTS = {
    "battle", "war", "siege", "revolt", "rebellion", "massacre", "treaty",
    "campaign", "conquest", "coronation", "expedition", "movement",
}

KING_TITLES = {"king", "emperor", "sultan", "maharaja", "raja", "nawab", "peshwa", "guru", "shah", "prince", "ruler"}

COMMON_STOP_PHRASES = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
    "according", "chapter", "volume", "page", "history", "historical"
}


def normalize_text(text: str) -> str:
    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def fuzzy_match(query: str, candidates: Iterable[str], threshold: float = 0.75) -> Optional[str]:
    query_n = normalize_name(query)
    best_match = None
    best_score = 0.0
    for cand in candidates:
        score = SequenceMatcher(None, query_n, normalize_name(cand)).ratio()
        if score > best_score:
            best_score = score
            best_match = cand
    if best_score >= threshold:
        return best_match
    return None


def detect_followup(history: List[Dict], new_question: str, threshold: float = 0.55) -> bool:
    if not history:
        return False
    tokens = set(re.findall(r"\w+", new_question.lower()))
    if tokens & PRONOUNS:
        return True
    return False


def best_phrase_candidate(text: str) -> str:
    chunks = re.findall(r"[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){0,4}", text)
    if not chunks:
        return ""
    chunks = [c.strip() for c in chunks if len(c.strip()) > 2]
    chunks.sort(key=len, reverse=True)
    return chunks[0] if chunks else ""


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in parts if p.strip()]


def score_query_intent(query: str) -> Dict[str, int]:
    q = query.lower()
    scores = {"graph": 0, "timeline": 0, "verify": 0, "vector": 0}
    graph_terms = [
        "who fought whom", "fought whom", "allied with", "succeeded by", "contemporary of",
        "relationship", "between", "connected to", "lineage", "descended from",
        "ruled by", "who ruled", "which king", "who was successor", "path", "chain"
    ]
    timeline_terms = [
        "when", "year", "date", "timeline", "chronology", "before", "after",
        "during", "period", "sequence", "in what year"
    ]
    verify_terms = ["compare", "compare historians", "according to", "viewpoint", "interpretation", "differentiate", "who says", "cross-check", "verify"]
    for t in graph_terms:
        if t in q:
            scores["graph"] += 2
    for t in timeline_terms:
        if t in q:
            scores["timeline"] += 1
    for t in verify_terms:
        if t in q:
            scores["verify"] += 2
    if len(q.split()) <= 8:
        scores["vector"] += 1
    if scores["verify"] == 0 and ("historian" in q or "majumdar" in q or "sarkar" in q):
        scores["verify"] += 2
    if scores["graph"] == 0 and ("fought" in q or "allied" in q or "succeeded" in q or "ruled" in q):
        scores["graph"] += 1
    return scores
