# phase_4_rlhf_faithfulness/self_correction.py
"""
Generate -> Critique -> Refine loop (STaR-style rationalization / Reflexion),
built on top of Phase 3's RAFTModel.

Phase 3's `RAFTModel.answer()` returns a single freeform pass. This wrapper
calls the *same* underlying model three times (or stops early once the
critique comes back clean):

    1. generate  -- draft answer with inline [n] citations (prompts.build_generation_prompt)
    2. critique  -- self-check against the documents (prompts.build_critique_prompt)
    3. refine    -- rewrite fixing flagged issues (prompts.build_refine_prompt)

The critique step is deliberately double-checked by `citation_utils.score_answer`
(a cheap, deterministic pass) rather than trusting the model's self-report
alone -- LLMs are notoriously charitable when grading their own citations.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from phase_3_moe_raft.raft_model import RAFTModel
from phase_4_rlhf_faithfulness import prompts, citation_utils
from phase_4_rlhf_faithfulness.config import MAX_REFINE_ROUNDS, FAITHFULNESS_ACCEPT_THRESHOLD


@dataclass
class CorrectionTrace:
    rounds: List[dict] = field(default_factory=list)   # [{draft, critique, score, problems}]
    final_answer: str = ""
    final_score: float = 0.0
    stopped_reason: str = ""


def _raw_generate(model: RAFTModel, prompt: str, max_new_tokens: int = 800) -> str:
    """Low-level single-turn generation reusing RAFTModel's loaded tokenizer/model."""
    messages = [
        {"role": "system", "content": "You are a precise, citation-disciplined historian assistant."},
        {"role": "user", "content": prompt},
    ]
    rendered = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(rendered, return_tensors="pt").to(model.model.device)
    outputs = model.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
    )
    full = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant\n" in full:
        return full.split("assistant\n")[-1].strip()
    return full[len(rendered):].strip()


class SelfCorrectingRAFTModel:
    def __init__(self, model: RAFTModel, embedder=None):
        self.model = model
        self.embedder = embedder  # optional shared.embeddings.EmbeddingModel().impl

    def answer(self, question: str, docs: List[str], domain: str = None,
               perspective: str = None, verbose: bool = False) -> CorrectionTrace:
        trace = CorrectionTrace()

        gen_prompt = prompts.build_generation_prompt(question, docs, domain, perspective)
        draft = _raw_generate(self.model, gen_prompt)

        for round_idx in range(MAX_REFINE_ROUNDS + 1):
            report = citation_utils.score_answer(draft, docs, embedder=self.embedder)
            trace.rounds.append({
                "round": round_idx,
                "draft": draft,
                "score": report.score,
                "problems": report.problems,
            })
            if verbose:
                print(f"[round {round_idx}] score={report.score:.2f} problems={report.problems}")

            if report.score >= FAITHFULNESS_ACCEPT_THRESHOLD:
                trace.stopped_reason = "faithfulness_threshold_met"
                break
            if round_idx == MAX_REFINE_ROUNDS:
                trace.stopped_reason = "max_rounds_reached"
                break

            # Self-critique: ask the model too (cheap, and useful for logging /
            # human review even though we gate the loop on the deterministic score).
            critique_prompt = prompts.build_critique_prompt(question, docs, draft)
            model_critique = _raw_generate(self.model, critique_prompt, max_new_tokens=800)
            combined_critique = model_critique + "\n" + citation_utils.problems_to_critique_text(report.problems)

            refine_prompt = prompts.build_refine_prompt(question, docs, draft, combined_critique)
            draft = _raw_generate(self.model, refine_prompt)

        trace.final_answer = trace.rounds[-1]["draft"]
        trace.final_score = trace.rounds[-1]["score"]
        return trace
