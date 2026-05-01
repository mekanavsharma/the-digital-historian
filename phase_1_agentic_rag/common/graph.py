"""Graph wiring for the phase 1 autonomous agent."""


from langgraph.graph import END, StateGraph

from .state import HistorianState


class Phase1GraphBuilder:
    def __init__(
        self,
        memory_tool,
        planner_tool,
        retrieval_tool,
        position_tool,
        claim_aligner_tool,
        evaluator_tool,
        synthesizer_tool,
    ):
        self.memory_tool = memory_tool
        self.planner_tool = planner_tool
        self.retrieval_tool = retrieval_tool
        self.position_tool = position_tool
        self.claim_aligner_tool = claim_aligner_tool
        self.evaluator_tool = evaluator_tool
        self.synthesizer_tool = synthesizer_tool

    def route_after_eval(self, state: HistorianState):

        if state.step_count >= state.max_steps:
            return "final_synthesizer"

        if state.evaluation and state.evaluation.get("needs_replan"):
            return "planner"

        return "final_synthesizer"

    def build(self):
        workflow = StateGraph(HistorianState)

        workflow.add_node("memory_manager", self.memory_tool.run)
        workflow.add_node("planner", self.planner_tool.run)
        workflow.add_node("retrieve_tool", self.retrieval_tool.run_lane)
        workflow.add_node("position_extractor", self.position_tool.run)
        workflow.add_node("claim_aligner", self.claim_aligner_tool.run)
        workflow.add_node("evaluator", self.evaluator_tool.run)
        workflow.add_node("final_synthesizer", self.synthesizer_tool.run)

        workflow.set_entry_point("memory_manager")
        workflow.add_edge("memory_manager", "planner")
        workflow.add_conditional_edges("planner", self.retrieval_tool.route)
        workflow.add_edge("retrieve_tool", "position_extractor")
        workflow.add_edge("position_extractor", "claim_aligner")
        workflow.add_edge("claim_aligner", "evaluator")
        workflow.add_conditional_edges("evaluator", self.route_after_eval)
        workflow.add_edge("final_synthesizer", END)

        return workflow.compile()
