# Phase 4 — Robustness & Reliability ("The Trust")

Builds on `phase_0_rag_baseline` (hybrid BM25+FAISS retrieval) and
`phase_3_moe_raft` (MoE-routed RAFT-tuned generator). Nothing here replaces
those — this phase wraps the Phase 3 model with a citation-faithfulness
training loop and a runtime self-correction loop, and scores everything with
RAGAS.

## Why the prompt format changed

Phase 3's `raft_model.py` never asked the model to cite anything inline —
Phase 0's `generate_answer()` just appended all retrieved `chunk_id`s at the
end, regardless of which ones actually supported which sentence. That's fine
for traceability but gives you nothing to optimize: you can't reward "cited
the right source" if the model was never asked to attribute claims to
specific sources in the first place.

`phase_4_rlhf_faithfulness/prompts.py` requires inline markers instead — `[1]`,
`[2]`, etc. referring to `### Document [n]` — so `citation_utils.py` can
check, per sentence, whether the citation (a) points at a real retrieved
document and (b) is actually about what the sentence claims.

## Files
- `prompts.py` — citation-aware prompts requiring inline [1], [2] markers tied to `### Document [n]`. This didn't exist before; `Phase 3`'s model never had to attribute claims to specific sources, which is why there was nothing to optimize.

- `citation_utils.py` — deterministic scorer: extracts cited indices per sentence, checks they're real retrieved docs, checks semantic support (embedding cosine, swappable for an NLI cross-encoder), flags uncited/fabricated claims. Tested it live above — works correctly.

- `self_correction.py` — the STaR/Reflection loop: generate → critique → refine, gated by the deterministic score (not just trusting the model's self-report).

- `preference_data_gen.py` — samples N candidates per question via your Phase 0 retriever, scores them, keeps best/worst as a DPO pair, skips low-contrast pairs.

- `train_dpo.py` — DPOTrainer continuing from your Phase 3 RAFT adapter (mirrors `train_raft.py`'s style/config pattern).

- `ragas_eval.py` — faithfulness/answer_relevancy/context_precision/context_recall, appends to a running comparison CSV across sft_only → dpo → dpo_selfcorrect.

- `run_query.py` — Phase 4 entry point, same CLI shape as Phase 3's.

- `raft_to_dpo_conversion.ipynb` - This is to convert raft formatted data to dpo formatted data so that we can finetune.


## Pipeline

```
                                   ┌─────────────────────────┐
   eval/retrieval_eval.csv ──────▶│ preference_data_gen.py   │
   (or any questions.csv)         │  - retrieve (Phase 0)     │
                                   │  - sample N candidates    │
                                   │  - score w/ citation_utils│
                                   │  - keep best/worst pair   │
                                   └───────────┬──────────────┘
                                               ▼
                                  data/dpo_preference.jsonl
                                               │
                                               ▼
                                       train_dpo.py
                                (LoRA continues from Phase 3's
                                 raft_finetuned adapter)
                                               │
                                               ▼
                                  dpo_finetuned/  (Phase 4 policy)
                                               │
                     ┌─────────────────────────┼──────────────────────┐
                     ▼                                                ▼
            run_query.py                                     ragas_eval.py
    (generate → self-critique →                        (faithfulness, answer_relevancy,
     refine, via self_correction.py)                    context_precision/recall)
                                                                        │
                                                                        ▼
                                                       rlhf_faithfulness/ragas_metrics.csv
```

## Running it end to end

```bash
# 1. Build preference data (smoke test first with --limit)
python -m phase_4_rlhf_faithfulness.preference_data_gen --limit 10
python -m phase_4_rlhf_faithfulness.preference_data_gen   # full run

# 2. DPO fine-tune on top of the Phase 3 adapter
python -m phase_4_rlhf_faithfulness.train_dpo

# 3. Query with self-correction
python -m phase_4_rlhf_faithfulness.run_query --query "Discuss Shah Jahan's architectural contributions."

# 4. Score all three configurations with RAGAS
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_3_moe_raft/raft_finetuned --tag sft_only
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo_selfcorrect --self-correct
```

## Design notes / things you'll likely need to tune

- **`_semantic_support` in `citation_utils.py`** currently falls back to
  lexical token overlap if you don't pass an embedder. For real training runs,
  pass `shared.embeddings.EmbeddingModel(model_name=...).impl` in — lexical
  overlap is only there so the scoring logic is testable without a GPU. On
  your 4GB 3050 this embedder call is cheap; the bottleneck is still
  generation.
- **DPO pair quality depends entirely on `CANDIDATES_PER_PROMPT` and
  `SAMPLING_TEMPERATURE`** (config.py). With only 4 candidates at temp 0.8 on
  a 3-4B model, expect a fair number of questions to get skipped for having
  too small a score gap (`MIN_SCORE_GAP`) — that's expected and fine, it's
  filtering out uninformative pairs, not a bug.
- **Training location**: given your 4GB VRAM constraint (same one that hit
  you during `train_raft.py`), do the actual DPO run on Kaggle T4s exactly
  like Phase 3 — use the local RTX 3050 only to smoke-test
  `preference_data_gen.py --limit 5` and `train_dpo.py` for a handful of
  steps to confirm no shape/pickling errors before shipping to Kaggle.
- **Self-correction cost**: each query now costs up to `1 + 2*MAX_REFINE_ROUNDS`
  forward passes instead of 1. `MAX_REFINE_ROUNDS=2` is a reasonable default;
  drop to 1 if latency matters more than the last bit of faithfulness.
- **RAGAS's LLM-based metrics** (`answer_relevancy`, parts of `faithfulness`)
  call an LLM internally (defaults to OpenAI unless you configure a RAGAS
  `LangchainLLMWrapper` around a local model). If you don't have an API key
  wired up, point RAGAS at your own Qwen model via its wrapper, or drop those
  two metrics and rely on `context_precision`/`context_recall` (retrieval-only,
  no extra LLM calls) plus `citation_utils.score_answer` for the generation
  side.

## Deliverables (`rlhf_faithfulness/`)

- `metrics_table.md` — template to fill in from `ragas_eval.py` runs.
- `before_after_examples.md` — template for hand-picked hallucination
  before/after comparisons.
- `ragas_metrics.csv` — auto-populated summary (one row per `--tag`).
- `ragas_per_question_<tag>.csv` — per-question RAGAS scores for digging into
  which question types still fail.



## DPO Fine Tuning Code
Execute it like this to launch both GPUs of Kaggle, It will be quicker.
This method is bit different than train_raft.py because DPO is inherently more GPU consuming than SFT(RAFT).

`!torchrun --standalone --nproc_per_node=2 /kaggle/input/datasets/kanav608/dpo-trainer/train_dpo.py`


## Generate DPO training data

run `raft_to_dpo_conversion.ipynb` using vllm in kaggle: loads `Qwen3-4B-Instruct-2507` in fp16 and, per chunk, generates a question + answer + distractor documents, saved as JSONL for `train_dpo.py` to consume.
Here is the structure of the file it follows:

```{
  "prompt": "You are a helpful assistant...\n\n### Document [1]: ...\n\n### Document [2]: ...\n\nQuestion: When did the Battle of Plassey take place?",
  "chosen": "1757",
  "rejected": "1857"
}
```