# phase_1/tools/final_synthesizer.py
"""
Synthesises a grounded, cited answer from gathered evidence and manages conversation memory.

LangGraph node – builds a grounded answer and updates chat history.
    Steps:
    1. Collects all evidence passages across positions.
    2. Builds a dynamic structural outline.
    3. Prompts the large LLM with strict grounding rules.
    4. Appends unique chunk IDs as inline citations.
    5. Updates conversation memory (summarising old turns if needed).

Args:
    hf_tuple:        (tokenizer, model) for the large generation model.
    small_hf_tuple:  (tokenizer, model) for the fast summarisation model.
    generate_answer_fn: function to generate answer from prompt and model.

Return a question-type-aware outline for the synthesizer prompt.

Matches on leading keywords / phrases to choose the most appropriate
section headings.
"""

from typing import Any, Dict, List
from phase_0_rag_baseline.config import RetrievalConfig

MAX_TURNS = RetrievalConfig.max_turns

class FinalSynthesizerTool:
    def __init__(self, hf_tuple, small_hf_tuple, generate_answer_fn):
        self.hf_tuple = hf_tuple
        self.small_hf_tuple = small_hf_tuple
        self.generate_answer = generate_answer_fn

    @staticmethod
    def build_outline(question: str) -> list[str]:
        q = question.lower()

        if any(q.startswith(w) for w in ["who", "which person", "which leader"]):
            return [
                "Direct answer",
                "Background and historical context",
                "Major actions or achievements",
                "Historical significance",
                "Conclusion",
            ]

        if any(k in q for k in ["why", "cause", "reason", "how did"]):
            return [
                "Historical context",
                "Key causes or developments",
                "Evidence from historical sources",
                "Interpretation and significance",
                "Conclusion",
            ]

        if any(k in q for k in ["what happened", "development", "event"]):
            return [
                "Background context",
                "Description of the event",
                "Key developments",
                "Historical consequences",
                "Conclusion",
            ]

        if any(k in q for k in ["role", "impact", "contribution", "importance"]):
            return [
                "Historical context",
                "Key actions or policies",
                "Impact on society or politics",
                "Long-term historical significance",
                "Conclusion",
            ]

        if any(k in q for k in ["compare", "difference", "versus", "vs"]):
            return [
                "Background context",
                "Comparison of key features",
                "Major differences",
                "Historical implications",
                "Conclusion",
            ]

        return [
            "Historical context",
            "Key developments",
            "Evidence and interpretation",
            "Historical significance",
            "Conclusion",
        ]

    def run(self, state) -> Dict[str, Any]:
        evidence_blocks = []
        all_chunk_ids: List[str] = []

        for lane_id, pos in state.positions.items():
            passages = pos.get("passages", [])
            chunk_ids = pos.get("chunk_ids", [])

            block_lines = [
                f"--- {lane_id.upper()} ---",
                f"Historian: {pos.get('historian', 'general')}",
                "",
                "Evidence Passages:",
            ]

            for i, passage in enumerate(passages):
                cid = chunk_ids[i] if i < len(chunk_ids) else "unknown"
                clean_text = passage.replace("\n", " ").strip()[:300]
                block_lines.append(f"[chunk_id={cid}] {clean_text}")
                all_chunk_ids.append(cid)

            evidence_blocks.append("\n".join(block_lines))

        evidence_text = "\n\n".join(evidence_blocks)
        if not evidence_text.strip():
            evidence_text = "No evidence retrieved."

        structure_list = self.build_outline(state.rewritten_query)
        structure_text = "\n".join(f"- {s}" for s in structure_list)

        prompt = f"""
    You are a **strictly grounded historian AI**.

    Use ONLY the evidence below.

    RULES:
    - No repetition
    - No hallucination
    - Every claim MUST cite [chunk_id=XYZ]
    - If insufficient evidence → say so briefly

    Question:
    {state.rewritten_query}

    Evidence:
    {evidence_text}

    Structure:
    {structure_text}

    Write {state.max_words or 650} words.
    """

        answer = self.generate_answer(
            prompt,
            self.hf_tuple,
            answer_style=state.answer_style,
            max_words=state.max_words or 650,
        )

        unique_chunks = list(dict.fromkeys([c for c in all_chunk_ids if c]))[:20]
        if unique_chunks:
            answer += "\n\n" + " ".join(f"[chunk_id={cid}]" for cid in unique_chunks)

        MAX_TURNS = 5
        history = state.chat_history or []

        if history and history[-1]["answer"] == "":
            history[-1]["answer"] = answer[:400]
        else:
            history.append({"question": state.original_query, "answer": answer[:400]})

        if len(history) > MAX_TURNS:
            old_history = history[:-MAX_TURNS]
            summary_text = "\n".join([f"Q: {h['question']} A: {h['answer']}" for h in old_history])
            summary_prompt = f"""
        Summarize the following conversation into key historical context.
        Keep important names, events, key ideas.
        Conversation:\n{summary_text}\nSummary:
        """
            summary = self.generate_answer(
                summary_prompt,
                self.small_hf_tuple,
                answer_style="very_short",
            )
            history = [{"question": "SUMMARY", "answer": summary}] + history[-MAX_TURNS:]

        return {
            "final_answer": answer,
            "chat_history": history,
        }
