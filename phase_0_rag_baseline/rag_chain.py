# phase_0_rag_baseline/rag_chain.py
from typing import Optional
from shared.prompts.rag_prompts import build_prompt
from phase_0_rag_baseline.llm import generate_answer

class RAGChain:
    def __init__(
        self,
        retriever,
        hf_tuple: Optional[tuple],
    ):
        self.retriever = retriever
        self.hf_tuple = hf_tuple

    def answer(
        self,
        query: str,
        cfg,
        answer_style: str = "concise",
        max_words: int = None,
    ) -> str:
        # max_words and max_new_tokens serve the same purpose to cap max no of tokens to be generated.
        # max_words is in prompt template It can be leveraged when we are setting answer_style = "detailed".
        candidates = self.retriever.retrieve(query, cfg)

        prompt, used_chunk_ids = build_prompt(
            candidates,
            query,
            answer_style=answer_style,
            max_words=max_words,
        )

        return generate_answer(prompt, self.hf_tuple, chunk_ids_used=used_chunk_ids,answer_style=answer_style, max_words=max_words)
