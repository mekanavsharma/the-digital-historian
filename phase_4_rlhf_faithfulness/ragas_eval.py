# phase_4_rlhf_faithfulness/ragas_eval.py
"""
Scores retrieval + generation quality with RAGAS, so you have an algorithmic
(not just vibes-based) before/after comparison across model versions:
    - Phase 3 base RAFT model (SFT only)
    - Phase 4 DPO model
    - Phase 4 DPO model + self-correction loop

Metrics used (all from ragas.metrics):
  - faithfulness       : is the answer supported by the retrieved contexts?
  - answer_relevancy   : does the answer actually address the question?
  - context_precision  : are the retrieved contexts relevant (ranked well)?
  - context_recall     : do the retrieved contexts cover the ground truth?

Usage:
    python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_3_moe_raft/raft_finetuned --tag sft_only
    python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo
    python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo_selfcorrect --self-correct

Appends a row per run to rlhf_faithfulness/ragas_metrics.csv so you can build
the final comparison table across all three configurations.
"""

import argparse
import os

import pandas as pd
from datasets import Dataset

from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever, Retriever
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig

from phase_3_moe_raft.raft_model import RAFTModel

from phase_4_rlhf_faithfulness import prompts
from phase_4_rlhf_faithfulness.self_correction import SelfCorrectingRAFTModel
from phase_4_rlhf_faithfulness.config import RAGAS_RESULTS_CSV, DELIVERABLE_DIR


def _build_retriever():
    retrieval_cfg = RetrievalConfig()
    model_cfg = ModelConfig()
    paths_cfg = PathsConfig()
    docs = load_jsonl_as_documents(paths_cfg.documents_path)
    bm25 = build_bm25_retriever(docs, bm25_path=paths_cfg.bm25_path)
    faiss_store = build_faiss_retriever(
        docs, model_name=model_cfg.embedding_model,
        index_path=paths_cfg.vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size,
    )
    reranker = load_cross_encoder(model_cfg.reranker_model)
    return Retriever(bm25=bm25, faiss=faiss_store, reranker=reranker), retrieval_cfg, paths_cfg


def build_ragas_dataset(eval_csv: str, model_path: str, top_k: int = 8,
                         self_correct: bool = False, limit: int = None) -> Dataset:
    retriever, retrieval_cfg, paths_cfg = _build_retriever()
    model = RAFTModel(model_path=model_path)
    wrapped = SelfCorrectingRAFTModel(model) if self_correct else None

    df = pd.read_csv(eval_csv)
    if limit:
        df = df.head(limit)

    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for _, row in df.iterrows():
        question = str(row["question"])
        ground_truth = str(row.get("correct_answer", ""))

        docs = retriever.retrieve(question, retrieval_cfg)[:top_k]
        doc_texts = [d.page_content for d in docs]
        if not doc_texts:
            continue

        if self_correct:
            trace = wrapped.answer(question, doc_texts)
            answer = trace.final_answer
        else:
            gen_prompt = prompts.build_generation_prompt(question, doc_texts)
            messages = [
                {"role": "system", "content": "You are a precise, citation-disciplined historian assistant."},
                {"role": "user", "content": gen_prompt},
            ]
            rendered = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = model.tokenizer(rendered, return_tensors="pt").to(model.model.device)
            outputs = model.model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                            pad_token_id=model.tokenizer.eos_token_id)
            full = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full.split("assistant\n")[-1].strip() if "assistant\n" in full else full[len(rendered):].strip()

        rows["question"].append(question)
        rows["answer"].append(answer)
        rows["contexts"].append(doc_texts)
        rows["ground_truth"].append(ground_truth)

    return Dataset.from_dict(rows)


def run_ragas(dataset: Dataset):
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return result.to_pandas()


def main():
    parser = argparse.ArgumentParser(description="RAGAS evaluation")
    parser.add_argument("--eval-csv", default="eval/retrieval_eval.csv")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tag", required=True, help="Label for this run, e.g. 'sft_only', 'dpo', 'dpo_selfcorrect'")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--self-correct", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ds = build_ragas_dataset(args.eval_csv, args.model_path, top_k=args.top_k,
                              self_correct=args.self_correct, limit=args.limit)
    per_row = run_ragas(ds)

    os.makedirs(DELIVERABLE_DIR, exist_ok=True)
    summary = per_row[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
    summary_row = pd.DataFrame([{"run_tag": args.tag, **summary.to_dict()}])

    if os.path.exists(RAGAS_RESULTS_CSV):
        existing = pd.read_csv(RAGAS_RESULTS_CSV)
        existing = existing[existing["run_tag"] != args.tag]  # replace if re-run
        combined = pd.concat([existing, summary_row], ignore_index=True)
    else:
        combined = summary_row

    combined.to_csv(RAGAS_RESULTS_CSV, index=False)
    per_row.to_csv(os.path.join(DELIVERABLE_DIR, f"ragas_per_question_{args.tag}.csv"), index=False)

    print(f"\n=== RAGAS summary ({args.tag}) ===")
    print(summary_row.to_string(index=False))
    print(f"\nAppended to {RAGAS_RESULTS_CSV}")


if __name__ == "__main__":
    main()
