# phase_4_rlhf_faithfulness/citation_utils.py
"""
Lightweight, model-free(ish) tools to measure "is this answer faithful to the
documents it cites". Two layers:

1. Structural validity (cheap, deterministic): does every cited [n] refer to a
   document index that was actually retrieved?
2. Semantic faithfulness (needs an NLI/entailment or embedding model): for each
   cited sentence, is the claim actually supported by the text of the document
   it cites?

We default to a sentence-embedding cosine-similarity proxy for (2), because it
has no extra GPU-heavy dependency beyond what Phase 0 already loads
(EmbeddingModel). If you have budget for a proper NLI cross-encoder
(e.g. "cross-encoder/nli-deberta-v3-base"), swap it in via
`load_nli_scorer()` -- it's a straight drop-in for `_semantic_support`.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_cited_indices(answer: str) -> List[int]:
    """All document indices (1-based) cited anywhere in the answer."""
    found = []
    for m in CITATION_RE.finditer(answer):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                found.append(int(tok))
    return found


def split_into_claims(answer: str) -> List[Tuple[str, List[int]]]:
    """
    Returns list of (sentence_text_without_citation, cited_indices) for every
    sentence in the answer. Sentences with no citation get an empty list.
    """
    claims = []
    for sent in SENTENCE_SPLIT_RE.split(answer.strip()):
        sent = sent.strip()
        if not sent:
            continue
        idxs = extract_cited_indices(sent)
        clean = CITATION_RE.sub("", sent).strip()
        if clean:
            claims.append((clean, idxs))
    return claims


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().startswith("i don't know") or answer.strip().lower().startswith("i dont know")


@dataclass
class FaithfulnessReport:
    citation_validity: float     # fraction of citations pointing at real doc indices
    faithfulness: float          # fraction of cited claims semantically supported
    uncited_claim_ratio: float   # fraction of factual sentences with NO citation at all
    abstained: bool
    problems: List[str]          # human-readable list, reused by the self-critique step

    @property
    def score(self) -> float:
        if self.abstained:
            return 1.0
        # Heavily penalize uncited claims -- an uncited "fact" is the classic
        # hallucination shape we're trying to train out.
        return max(
            0.0,
            0.5 * self.citation_validity + 0.5 * self.faithfulness - 0.5 * self.uncited_claim_ratio,
        )


def _semantic_support(claim: str, doc_text: str, embedder=None, threshold: float = 0.55) -> bool:
    """
    Cosine-similarity proxy for entailment. `embedder` should expose
    `.embed_query(text) -> List[float]` (matches shared.embeddings.EmbeddingModel's
    underlying LangChain-style interface). Falls back to a crude lexical
    overlap check if no embedder is supplied (useful for fast unit tests).
    """
    if embedder is None:
        claim_tokens = set(w.lower() for w in re.findall(r"\w+", claim) if len(w) > 3)
        doc_tokens = set(w.lower() for w in re.findall(r"\w+", doc_text) if len(w) > 3)
        if not claim_tokens:
            return True
        overlap = len(claim_tokens & doc_tokens) / len(claim_tokens)
        return overlap >= 0.35

    import numpy as np
    a = np.array(embedder.embed_query(claim))
    b = np.array(embedder.embed_query(doc_text[:2000]))
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return False
    sim = float(np.dot(a, b) / denom)
    return sim >= threshold


def score_answer(answer: str, retrieved_docs: List[str], embedder=None) -> FaithfulnessReport:
    """
    retrieved_docs: list of raw document text, in the SAME order they were
    numbered in the prompt (index 0 -> "Document [1]", etc).
    """
    if is_abstention(answer):
        return FaithfulnessReport(1.0, 1.0, 0.0, True, [])

    n_docs = len(retrieved_docs)
    claims = split_into_claims(answer)
    if not claims:
        return FaithfulnessReport(0.0, 0.0, 1.0, False, ["No factual sentences parsed from answer."])

    problems = []
    valid_citations = 0
    total_citations = 0
    supported_claims = 0
    cited_claims = 0
    uncited_claims = 0

    for clean_sent, idxs in claims:
        if not idxs:
            uncited_claims += 1
            problems.append(f"Uncited claim: \"{clean_sent[:100]}\"")
            continue
        cited_claims += 1
        sentence_supported = False
        for i in idxs:
            total_citations += 1
            if 1 <= i <= n_docs:
                valid_citations += 1
                if _semantic_support(clean_sent, retrieved_docs[i - 1], embedder=embedder):
                    sentence_supported = True
            else:
                problems.append(f"Citation [{i}] does not correspond to any retrieved document.")
        if sentence_supported:
            supported_claims += 1
        else:
            problems.append(f"Unsupported claim despite citation: \"{clean_sent[:100]}\"")

    citation_validity = valid_citations / total_citations if total_citations else 0.0
    faithfulness = supported_claims / cited_claims if cited_claims else 0.0
    uncited_ratio = uncited_claims / len(claims)

    return FaithfulnessReport(
        citation_validity=citation_validity,
        faithfulness=faithfulness,
        uncited_claim_ratio=uncited_ratio,
        abstained=False,
        problems=problems,
    )


def problems_to_critique_text(problems: List[str]) -> str:
    if not problems:
        return "No issues found."
    return "\n".join(f"- {p}" for p in problems[:8])
