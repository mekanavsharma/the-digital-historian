# phase_4_rlhf_faithfulness/preference_data_gen.py
"""
Builds a DPO preference dataset (chosen/rejected pairs) from your existing
Phase 0 hybrid retriever + Phase 3 RAFT model.

For every question in eval/retrieval_eval.csv (or a custom jsonl of
questions), we:
  1. Retrieve docs the same way phase_3's run_query.py does.
  2. Sample CANDIDATES_PER_PROMPT completions at temperature > 0, using the
     citation-aware prompt from prompts.build_generation_prompt.
  3. Score each candidate with citation_utils.score_answer.
  4. Take the highest-scoring candidate as `chosen` and the lowest-scoring as
     `rejected`. Skip the pair if the score gap is too small to be a useful
     training signal (< 0.1) -- that pair wouldn't teach the model anything
     and just adds noise.

Output: phase_4_rlhf_faithfulness/data/dpo_preference.jsonl, one JSON object per
line with keys: prompt, chosen, rejected, chosen_score, rejected_score.
This is the format trl.DPOTrainer expects (prompt/chosen/rejected columns).
"""

import argparse
import json
import os

import pandas as pd

from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever, Retriever
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig

from phase_3_moe_raft.raft_model import RAFTModel
from phase_3_moe_raft.config import RAFT_MODEL_PATH

from phase_4_rlhf_faithfulness import prompts, citation_utils
from phase_4_rlhf_faithfulness.config import (
    PREFERENCE_DATA_PATH, CANDIDATES_PER_PROMPT, SAMPLING_TEMPERATURE, SAMPLING_TOP_P,
    MAX_NEW_TOKENS,
)

MIN_SCORE_GAP = 0.1


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
    return Retriever(bm25=bm25, faiss=faiss_store, reranker=reranker), retrieval_cfg


def _sample_candidates(model: RAFTModel, gen_prompt: str, n: int) -> list:
    messages = [
        {"role": "system", "content": "You are a precise, citation-disciplined historian assistant."},
        {"role": "user", "content": gen_prompt},
    ]
    rendered = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = model.tokenizer(rendered, return_tensors="pt").to(model.model.device)

    candidates = []
    for _ in range(n):
        outputs = model.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=SAMPLING_TEMPERATURE,
            top_p=SAMPLING_TOP_P,
            pad_token_id=model.tokenizer.eos_token_id,
        )
        full = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = full.split("assistant\n")[-1].strip() if "assistant\n" in full else full[len(rendered):].strip()
        candidates.append(text)
    return candidates


def build_preference_dataset(questions_csv: str, top_k: int = 8, limit: int = None):
    retriever, retrieval_cfg = _build_retriever()
    model = RAFTModel(model_path=RAFT_MODEL_PATH)

    df = pd.read_csv(questions_csv)
    if limit:
        df = df.head(limit)

    os.makedirs(os.path.dirname(PREFERENCE_DATA_PATH), exist_ok=True)
    n_written = 0

    with open(PREFERENCE_DATA_PATH, "w", encoding="utf-8") as f_out:
        for _, row in df.iterrows():
            question = str(row["question"])
            docs = retriever.retrieve(question, retrieval_cfg)[:top_k]
            doc_texts = [d.page_content for d in docs]
            if not doc_texts:
                continue

            gen_prompt = prompts.build_generation_prompt(question, doc_texts)
            candidates = _sample_candidates(model, gen_prompt, CANDIDATES_PER_PROMPT)

            scored = []
            for cand in candidates:
                report = citation_utils.score_answer(cand, doc_texts)
                scored.append((report.score, cand))
            scored.sort(key=lambda x: x[0], reverse=True)

            best_score, best_answer = scored[0]
            worst_score, worst_answer = scored[-1]

            if best_score - worst_score < MIN_SCORE_GAP or best_answer == worst_answer:
                continue  # not a useful preference signal

            record = {
                "prompt": gen_prompt,
                "chosen": best_answer,
                "rejected": worst_answer,
                "chosen_score": best_score,
                "rejected_score": worst_score,
                "question": question,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} preference pairs to {PREFERENCE_DATA_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build DPO preference dataset for citation faithfulness")
    parser.add_argument("--questions-csv", default="eval/retrieval_eval.csv",
                        help="CSV with a 'question' column")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of questions (for smoke tests)")
    args = parser.parse_args()
    build_preference_dataset(args.questions_csv, top_k=args.top_k, limit=args.limit)


if __name__ == "__main__":
    main()
