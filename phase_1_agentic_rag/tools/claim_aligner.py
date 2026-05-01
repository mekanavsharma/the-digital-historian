# phase_1/tools/claim_aligner.py

"""
Optional node that compares historian viewpoints across lanes to surface agreements and disagreements.

Ask the small LLM to compare historian claims across lanes and return a
structured dict of agreements, disagreements, and unique interpretations.

Args:
    lane_results:    List of lane dicts (each with "historian" and "positions" keys).
    query:           The rewritten user query.
    small_hf_tuple:  (tokenizer, model) for the fast routing model.

Returns:
    Dict with keys:
        comparison    – {"agreements": [], "disagreements": [], "unique_views": []}
        per_historian – [{"historian": ..., "claims": [...]}, ...]
"""


"""Claim alignment node."""

import json
from typing import Dict, List


class ClaimAlignerTool:
    def __init__(self, small_hf_tuple, generate_answer_fn):
        self.small_hf_tuple = small_hf_tuple
        self.generate_answer = generate_answer_fn

    def claim_alignment(self, lane_results: List[Dict], query: str) -> Dict:
        historian_claims = []

        for lane in lane_results:
            historian = lane["historian"]
            claims = []

            for pos in lane.get("positions", []):
                claims.extend([c["claim"] for c in pos.get("claims", [])])

            historian_claims.append(
                {
                    "historian": historian,
                    "claims": claims,
                }
            )

        claims_text = "\n".join(
            [
                f"{h['historian']}: {', '.join(h['claims'])}"
                for h in historian_claims
            ]
        )

        prompt = f"""
            You are comparing historical interpretations.

            Query:
            {query}

            Claims by historians:
            {claims_text}

            Identify:

            1. Agreements between historians
            2. Disagreements between historians
            3. Unique interpretations

            Return JSON:

            {{
             "agreements": [],
             "disagreements": [],
             "unique_views": []
            }}
            """

        try:
            result = json.loads(
                self.generate_answer(
                    prompt,
                    self.small_hf_tuple,
                    answer_style="very_short",
                )
            )
        except Exception:
            result = {"agreements": [], "disagreements": [], "unique_views": []}

        return {
            "comparison": result,
            "per_historian": historian_claims,
        }

    def run(self, state):
        if not state.plan.get("use_claim_aligner", False):
            return {"claim_comparison": None}
        result = self.claim_alignment(state.lanes, state.rewritten_query)
        return {"claim_comparison": result.get("comparison")}
