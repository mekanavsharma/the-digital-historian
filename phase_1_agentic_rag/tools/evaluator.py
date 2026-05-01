# phase_1/tools/evaluator.py

"""
Deterministic evidence-sufficiency check.

If too few evidence passages were retrieved the evaluator signals the graph
to loop back to the planner (needs_replan=True).  Once evidence is sufficient
– or the step budget is exhausted – it routes to the final synthesizer.

Returns:
    Dict with keys: evaluation (dict), and optionally step_count.
"""

from typing import Any, Dict

class EvaluatorTool:
    def run(self, state) -> Dict[str, Any]:
        evidence_count = 0

        for pos in state.positions.values():
            evidence_count += len(pos.get("passages", []))


        # if evidence_count < 2 and state.step_count < state.max_steps:
        max_replans = 2

        if evidence_count < 2 and state.step_count < max_replans:
            return {
                "evaluation": {
                    "needs_replan": True,
                    "reason": "insufficient evidence",
                },
                "step_count": state.step_count + 1,
            }

        return {
            "evaluation": {
                "needs_replan": False,
                "reason": "evidence sufficient",
            }
        }
