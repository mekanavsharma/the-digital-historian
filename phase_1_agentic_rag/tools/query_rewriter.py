# phase_1/tools/query_rewriter.py

"""
Resolve pronouns and vague references in follow-up queries using the conversation history.

Use the small (fast) LLM to resolve any pronouns or vague references in
*query* by looking at recent conversation *history*.

If history is empty the original query is returned unchanged.

Args:
    query:           The user's raw question (may contain pronouns).
    history:         List of {"question": ..., "answer": ...} dicts.
    small_hf_tuple:  (tokenizer, model) for the small routing model.

Returns:
    Rewritten question string.  Falls back to *query* if the model produces an unusable result.
"""


from typing import List

class QueryRewriterTool:
    def __init__(self, small_hf_tuple, generate_answer_fn):
        self.small_hf_tuple = small_hf_tuple
        self.generate_answer = generate_answer_fn

    def run(self, query: str, history: List[dict]) -> str:
        if not history:
            return query

        history_text = "\n".join(
            [f"Q: {h['question']}\nA: {h['answer'][:200]}" for h in history[-3:]]
        )

        prompt = f"""You are rewriting a historical research query.
        Your task: resolve any pronouns or vague references using the conversation history.

        Conversation history:
        {history_text}

        Current question:
        {query}

        Rules:
        - If the question uses "he/she/they/his/it/this", replace with the specific person or topic from history.
        - If the question is self-contained, return it unchanged.
        - Output ONLY the rewritten question, nothing else.

        Rewritten question:"""

        result = self.generate_answer(
            prompt,
            self.small_hf_tuple,
            answer_style="very_short",
        ).strip()
        return result if len(result) > 5 else query
