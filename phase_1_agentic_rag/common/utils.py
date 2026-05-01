"""Shared helpers for the phase 1 graph."""

from typing import Dict, Optional

import numpy as np


def merge_dicts(left: Optional[Dict], right: Optional[Dict]) -> Dict:
    left = left or {}
    right = right or {}
    merged = dict(left)
    merged.update(right)
    return merged


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


PRONOUNS = {"he", "she", "they", "his", "her", "their", "it", "its",
            "him", "this", "that", "these", "those", "the same"}


def detect_followup(history, new_question, embed_model=None, threshold: float = 0.55):
    if not history:
        return False

    tokens = set(new_question.lower().split())
    if tokens & PRONOUNS:
        return True

    if embed_model is None:
        return False

    last_q = history[-1]["question"]
    q1 = embed_model.encode(last_q)
    q2 = embed_model.encode(new_question)
    sim = cosine_sim(q1, q2)
    return sim > threshold


def normalize_ocr_text(text: str) -> str:
    import re

    text = re.sub(r"-\n\s*", "", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
