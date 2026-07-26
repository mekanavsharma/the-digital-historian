# phase_4_trust_rlhf/prompts.py
"""
Citation-aware prompts.

Unlike Phase 3's raft_model.build_system_prompt (which asks for a freeform
answer with citations bolted on afterwards), Phase 4 requires the model to
place an inline [n] marker after every claim, where n refers to the
"### Document [n]" blocks it was given. This is what makes citation
faithfulness measurable and optimizable (DPO reward, RAGAS, self-critique).
"""

from typing import List, Tuple


def build_numbered_context(docs: List[str]) -> str:
    """docs: list of raw document text (already truncated by the caller)."""
    parts = [f"### Document [{i + 1}]:\n{d}" for i, d in enumerate(docs)]
    return "\n\n".join(parts)


def build_generation_prompt(question: str, docs: List[str], domain: str = None,
                             perspective: str = None) -> str:
    style_hint = ""
    if domain and perspective:
        style_hint = f"Answer from a {perspective} historian's viewpoint, focused on the {domain} period. "

    docs_str = build_numbered_context(docs)
    return (
        f"{style_hint}"
        "You are a careful historian assistant. Use ONLY the numbered DOCUMENTS below.\n\n"
        f"DOCUMENTS:\n{docs_str}\n\n"
        f"QUESTION: {question}\n\n"
        "RULES:\n"
        "1. Every factual sentence must end with a citation marker like [1] or [2,3] "
        "referring to the Document numbers above that support it.\n"
        "2. Do not cite a document number that was not provided.\n"
        "3. If none of the documents answer the question, reply exactly: \"I don't know.\" "
        "(no citation needed in that case).\n"
        "4. Do not invent facts that are not present in the cited document(s).\n\n"
        "ANSWER:"
    )


def build_critique_prompt(question: str, docs: List[str], draft_answer: str) -> str:
    """Asks the model to check its own draft against the documents (Reflection step)."""
    docs_str = build_numbered_context(docs)
    return (
        "You are a strict fact-checking historian reviewing a draft answer.\n\n"
        f"DOCUMENTS:\n{docs_str}\n\n"
        f"QUESTION: {question}\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        "Check the draft against the documents. For each sentence in the draft ask:\n"
        "- Does it cite a real Document number that was actually provided?\n"
        "- Is the cited Document's text actually about what the sentence claims?\n"
        "- Is there any claim with no citation, or a citation to a fabricated/irrelevant source?\n\n"
        "Respond with a short critique (max 4 bullet points) listing only the problems found. "
        "If there are no problems, respond with exactly: \"No issues found.\""
    )


def build_refine_prompt(question: str, docs: List[str], draft_answer: str, critique: str) -> str:
    docs_str = build_numbered_context(docs)
    return (
        "You are a careful historian assistant revising your own answer after fact-checking.\n\n"
        f"DOCUMENTS:\n{docs_str}\n\n"
        f"QUESTION: {question}\n\n"
        f"DRAFT ANSWER:\n{draft_answer}\n\n"
        f"CRITIQUE OF DRAFT:\n{critique}\n\n"
        "Rewrite the answer to fix every problem raised in the critique. "
        "Keep the same citation rules: every factual sentence ends with [n] referring to a "
        "real Document number above, and if nothing supports the answer, say \"I don't know.\"\n\n"
        "REVISED ANSWER:"
    )
