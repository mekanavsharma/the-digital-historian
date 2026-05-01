# phase_1/tools/planner.py

"""
Plans retrieval lanes for the historian agent.

The planner uses a small, fast LLM to decompose the user query into
sub-queries and historian filters, then packages them into "lanes" that
the parallel retrieve_node will fan out via LangGraph's Send API.

LangGraph node – rewrites the query, generates a JSON plan, and builds
retrieval lanes.

Args:
    state:           Current HistorianState.
    small_hf_tuple:  (tokenizer, model) for the fast routing model.

Returns:
    Dict with keys: rewritten_query, plan, lanes, step_count.
"""

import json
from typing import Any, Dict

from phase_1_agentic_rag.tools.query_rewriter import QueryRewriterTool


class PlannerTool:
    def __init__(self, small_hf_tuple, generate_answer_fn):
        self.rewriter = QueryRewriterTool(small_hf_tuple, generate_answer_fn)
        self.generate_answer = generate_answer_fn
        self.small_hf_tuple = small_hf_tuple

    def run(self, state) -> Dict[str, Any]:
        rewritten = self.rewriter.run(state.original_query, state.chat_history)

        prompt = f"""You are an expert history research planner.
        Original user question: {state.original_query}
        Rewritten (with conversation context): {rewritten}

        Output ONLY valid JSON:
        {{
        "rewritten_query": "{rewritten}",
        "historians": ["general" or specific names, max 4],
        "sub_queries": ["focused sub-question 1", "focused sub-question 2"],
        "answer_style": "detailed",
        "max_words": 500,
        "use_claim_aligner": false,
        "needs_extra_retrieval": true,
        "reason": "one short sentence"
        }}"""

        plan_str = self.generate_answer(
            prompt,
            self.small_hf_tuple,
            answer_style="very_short",
        )

        try:
            plan = json.loads(plan_str)
        except Exception:
            plan = {
                "rewritten_query": rewritten,
                "historians": ["general"],
                "sub_queries": [rewritten],
                "answer_style": "detailed",
                "max_words": 700,
                "use_claim_aligner": False,
                "needs_extra_retrieval": True,
                "reason": "default",
            }

        # PRIORITY: user override > LLM > default
        if state.historian_override:
            historians = state.historian_override
        else:
            historians = plan.get("historians", ["general"])

        sub_queries = plan.get("sub_queries", [rewritten])

        if len(sub_queries) == 1 and len(historians) > 1:
            sub_queries = sub_queries * len(historians)
        elif len(sub_queries) != len(historians):
            sub_queries = [rewritten] * len(historians)

        lanes = [
            {"lane_id": f"lane_{i}", "historian": h, "question": q}
            for i, (h, q) in enumerate(zip(historians, sub_queries))
        ]

        return {
            "rewritten_query": plan["rewritten_query"],
            "plan": plan,
            "lanes": lanes,
            "step_count": state.step_count + 1,
        }
