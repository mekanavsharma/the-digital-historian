# phase_1/tools/memory_manager.py
"""
Return True if *new_question* is a follow-up to the last turn in *history*.

A question is deemed a follow-up when:
- It contains a pronoun from the PRONOUNS set (strong signal), OR
- Its cosine similarity with the previous question exceeds *threshold*.

An empty history always returns False (treat as a fresh query).
"""

from typing import Any, Dict
from phase_1_agentic_rag.common.utils import detect_followup
from shared.embeddings.embedder import EmbeddingModel

embedder = EmbeddingModel(encode_kwargs={})

class MemoryManagerTool:
    def __init__(self, embed_model):
        self.embed_model = embed_model

    def run(self, state) -> Dict[str, Any]:
        history = state.chat_history or []
        is_followup = detect_followup(
            history,
            state.original_query,
            embed_model=self.embed_model,
        )
        print(f"Is follow-up: {is_followup}")
        if not is_followup:
            history = []
        history.append({"question": state.original_query, "answer": ""})
        return {"chat_history": history}
