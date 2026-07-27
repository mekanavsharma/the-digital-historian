# Phase 4 — RLHF & Faithfulness ("The Trust")

Phase 4 builds on `phase_0_rag_baseline` (hybrid retrieval) and `phase_3_moe_raft`
(MoE-routed RAFT model). It doesn't replace either — it wraps the Phase 3 model
with a citation-faithfulness training loop (DPO), a runtime self-correction
loop, and RAGAS-based evaluation, so answers can be scored and improved on
"did it cite the right source", not just "did it sound right".


## What this phase does

- Builds **DPO preference pairs** by sampling multiple candidate answers per question and scoring them on citation validity + faithfulness
- **DPO fine-tunes** on top of the Phase 3 RAFT adapter to reward faithful, well-cited answers over unfaithful ones
- Wraps generation in a **self-correction loop** (generate → critique → refine) at query time, gated by a deterministic faithfulness score
- Evaluates all three stages (SFT-only → DPO → DPO + self-correct) with **RAGAS**

## Folder layout

```text
phase_4_rlhf_faithfulness/
├── config.py                     # paths, DPO hyperparams, self-correction thresholds
├── prompts.py                    # citation-aware prompts requiring inline [n] markers
├── citation_utils.py              # scores an answer: citation validity + semantic faithfulness
├── preference_data_gen.py        # samples N candidates/question, scores, keeps best/worst DPO pair
├── train_dpo.py                  # DPOTrainer, continues from Phase 3's RAFT adapter
├── self_correction.py            # generate → critique → refine loop (STaR/Reflection)
├── ragas_eval.py                  # faithfulness/relevancy/precision/recall, appends to a run CSV
├── run_query.py                  # CLI entry point
├── raft_to_dpo_conversion.ipynb   # Kaggle notebook — generates DPO preference data
└── dpo_finetuned/                 # output of train_dpo.py — LoRA adapter + tokenizer
```

## Runtime flow

```text
USER QUERY
    ↓
router.py (Phase 3)   → classify_domain() + classify_perspective()
    ↓
ExpertRetriever (Phase 3) → phase 0 hybrid retrieve, filtered to matching expert
    ↓
prompts.py            → build citation-aware prompt (inline [n] markers)
    ↓
SelfCorrectingRAFTModel:
    generate  → citation_utils.score_answer()
    critique  → refine   (repeat up to MAX_REFINE_ROUNDS, or until score ≥ threshold)
    ↓
Final answer (DPO-tuned model, self-corrected)
```

**Training-side flow** (offline, produces the model used above):

```text
questions ──► preference_data_gen.py ──► dpo_preference.jsonl ──► train_dpo.py ──► dpo_finetuned/
              (sample N candidates,                              (LoRA continues
               score, keep best/worst)                            from Phase 3 adapter)
```

## How this phase reuses earlier code

**Phase 0**
- `phase_0_rag_baseline/retriever.py`, `ingest.py`, `reranker.py` — same hybrid retrieval stack `run_query.py` builds on

**Phase 3**
- `phase_3_moe_raft/router.py` — same query routing (domain + perspective)
- `phase_3_moe_raft/retriever.py` (`ExpertRetriever`) — same expert-filtered retrieval
- `phase_3_moe_raft/raft_model.py` (`RAFTModel`) — same model-loading class; Phase 4 wraps it in `SelfCorrectingRAFTModel` rather than replacing it
- `phase_3_moe_raft/raft_finetuned/` — the RAFT adapter is the starting checkpoint `train_dpo.py` continues from (`SFT_MODEL_PATH` in `config.py`)

Everything else (citation-aware prompts, scoring, DPO training, self-correction) is new in Phase 4.

## Models used

| Role | Model |
|---|---|
| DPO starting checkpoint | Phase 3's RAFT adapter (`phase_3_moe_raft/raft_finetuned`) |
| Base LLM (for comparison / `--model-path` override) | `Qwen/Qwen3-4B-Instruct-2507` |
| DPO training-data generator (Kaggle only) | `Qwen/Qwen2.5-1.5B-Instruct` |

## Running it end to end

```bash
# 1. Build preference data (smoke test first)
python -m phase_4_rlhf_faithfulness.preference_data_gen --limit 10
python -m phase_4_rlhf_faithfulness.preference_data_gen        # full run

# 2. DPO fine-tune on top of the Phase 3 adapter
python -m phase_4_rlhf_faithfulness.train_dpo

# 3. Query with self-correction
python -m phase_4_rlhf_faithfulness.run_query --query "Discuss Shah Jahan's architectural contributions."

# 4. Score all three configurations with RAGAS
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_3_moe_raft/raft_finetuned --tag sft_only
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo
python -m phase_4_rlhf_faithfulness.ragas_eval --model-path phase_4_rlhf_faithfulness/dpo_finetuned --tag dpo_selfcorrect --self-correct
```

Other `run_query.py` flags: `--domain`, `--perspective` (force instead of auto-route), `--no-self-correct` (first draft only), `--verbose` (print every refine round).

### Multi-GPU DPO on Kaggle

DPO is heavier than the RAFT/SFT run, so launch it across both Kaggle T4s:

```bash
!torchrun --standalone --nproc_per_node=2 /kaggle/input/datasets/kanav608/dpo-trainer/train_dpo.py
```

### Generate DPO training data

Run `raft_to_dpo_conversion.ipynb` on Kaggle (uses vLLM + `Qwen3-4B-Instruct-2507`) to turn source chunks into DPO pairs, saved as JSONL:

```json
{
  "prompt": "You are a helpful assistant...\n\n### Document [1]: ...\n\n### Document [2]: ...\n\nQuestion: When did the Battle of Plassey take place?",
  "chosen": "1757",
  "rejected": "1857"
}
```

## Notes

- **DPO pair quality depends on `CANDIDATES_PER_PROMPT` and `SAMPLING_TEMPERATURE`** (`config.py`). With only 4 candidates at temp 0.8, expect a fair number of questions to get skipped for too small a score gap (`MIN_SCORE_GAP`) — that's the filter working, not a bug.
- **`_semantic_support` in `citation_utils.py`** falls back to lexical token overlap if no embedder is passed. For real runs, pass `shared.embeddings.EmbeddingModel(...)` in — lexical overlap is only there so scoring is testable without a GPU.
- **Train on Kaggle T4s**, same as Phase 3 — use the local RTX 3050 only to smoke-test `preference_data_gen.py --limit 5` and a handful of `train_dpo.py` steps to catch shape/pickling errors first.
- **Self-correction cost**: each query costs up to `1 + 2*MAX_REFINE_ROUNDS` forward passes instead of 1. `MAX_REFINE_ROUNDS=2` is the default; drop to 1 if latency matters more than the last bit of faithfulness.
- **RAGAS's LLM-based metrics** (`answer_relevancy`, parts of `faithfulness`) call an LLM internally — defaults to OpenAI unless you wrap a local model via RAGAS's `LangchainLLMWrapper`. Without an API key wired up, either point RAGAS at your own Qwen model or drop those two metrics and rely on `context_precision`/`context_recall` plus `citation_utils.score_answer`.
- `ragas` isn't in the root `requirements.txt` yet — install it separately (`pip install ragas`) before running `ragas_eval.py`.
