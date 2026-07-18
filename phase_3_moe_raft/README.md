# Phase 3 — MoE & RAFT

Phase 3 adds two things on top of the phase 0 retrieval stack:

- **MoE-style routing** — every query is routed to one of 9 "experts" (3 time domains × 3 historian perspectives), and retrieval is filtered down to only that expert's documents.
- **RAFT (Retrieval-Augmented Fine-Tuning)** — a small instruct model, QLoRA fine-tuned to answer strictly from the retrieved documents and ignore distractors, instead of relying on parametric knowledge.

The same CLI (`run_query.py`) can answer with either the base LLM (zero-shot) or the fine-tuned RAFT model, switched with a single flag.

## What this phase does

- Uses a **keyword-based router** to classify a query into `expert_domain` (Ancient / Medieval / Modern) and `historian_perspective` (Nationalist / Marxist / Neutral)
- Uses `ExpertRetriever` to wrap the **phase 0 hybrid retriever** and post-filter results to the matched expert's chunks
- Loads either the **base LLM** (`Qwen3-4B-Instruct`, zero-shot) or the **RAFT fine-tuned adapter** (`Qwen2.5-3B-Instruct` + LoRA), detected automatically from the model path
- Includes the **training script** used to produce the RAFT adapter, and the **Kaggle notebook** used to generate its training data

## Folder layout

```text
phase_3_moe_raft/
├── config.py                     # paths, model paths, expert definitions
├── router.py                     # keyword classifier → (domain, perspective)
├── retriever.py                  # ExpertRetriever, filters phase 0 results by expert
├── raft_model.py                 # loads base model or PEFT adapter, generates answers
├── run_query.py                  # CLI entry point
├── train_raft.py                 # QLoRA fine-tuning script
├── build_metadata.py             # aggregates raw chunks into chunks_meta.json
├── raft_data_conversion.ipynb    # notebook — generates RAFT training data from existing json file
└── raft_finetuned/               # output of train_raft.py — LoRA adapter + tokenizer
```

## Runtime flow

```text
USER QUERY
    ↓
router.py            → classify_domain() + classify_perspective()
    ↓
ExpertRetriever       → phase 0 hybrid retrieve (BM25 + FAISS + reranker)
                        → filter to matching expert_domain / historian_perspective
    ↓
RAFTModel             → build system prompt for (domain, perspective)
                        → feed retrieved docs + question
    ↓
Base LLM  (Qwen3-4B-Instruct)      OR      RAFT adapter (Qwen2.5-3B-Instruct + LoRA)
    ↓
Answer
```

## How this phase reuses earlier code

**Phase 0**
- `phase_0_rag_baseline/retriever.py` provides the hybrid BM25 + FAISS + reranker stack that `ExpertRetriever` wraps
- `phase_0_rag_baseline/ingest.py` loads the JSONL corpus into documents

Everything else (routing, expert filtering, RAFT model loading, training) is new in phase 3.

## Setup — RAFT fine-tuning environment

Fine-tuning uses its own virtual environment, separate from the root `requirements.txt`:

```bash
python -m venv raft_env
raft_env\Scripts\activate        # Windows

pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

pip install transformers==4.44.0 tokenizers==0.19.1 accelerate==0.33.0 peft==0.12.0 trl==0.11.0 datasets==2.19.1 bitsandbytes==0.49.2 huggingface-hub==0.36.0
```

**Hardware:** local training was smoke-tested on an RTX 3050 Laptop GPU (4GB VRAM) just to confirm the script runs end-to-end. The real training run happens on Kaggle T4s — 4GB isn't enough for a full run.

## Models used

| Role | Model |
|---|---|
| Base LLM (MoE / zero-shot) | `Qwen/Qwen3-4B-Instruct-2507` |
| RAFT fine-tuning base | `Qwen/Qwen2.5-3B-Instruct` |

## Build the expert index (if not already built)

```bash
python -m phase_3_moe_raft.build_metadata

```

## Ask a question

```bash
# Base model (no fine-tuning), compare all 3 perspectives, show retrieved docs
python -m phase_3_moe_raft.run_query --query "What was the significance of Gandhi's Non-Cooperation Movement?" --compare --show-docs

# RAFT fine-tuned model
python -m phase_3_moe_raft.run_query --query "Why was Bengal partitioned?" --compare --raft-model

# Explicit model/checkpoint path (overrides --raft-model)
python -m phase_3_moe_raft.run_query --query "What was the economic drain?" --compare --model-path phase_3_moe_raft/raft_finetuned
```

Other flags: `--domain`, `--perspective` (force instead of auto-route), `--no-model` (retrieval only), `--top-k`.

## Fine-tune the RAFT model

```bash
python -m phase_3_moe_raft.train_raft
```

Reads `RAFT_TRAIN_JSON` from `config.py`, trains QLoRA on `Qwen2.5-1.5B-Instruct` (LoRA r=16, alpha=32, targeting all attention + MLP projections), and saves the adapter to `phase_3_moe_raft/raft_finetuned`.

## Generate RAFT training data (optional)

`raft_data_conversion.ipynb` runs on Kaggle: loads `Qwen3-4B-Instruct-2507` in fp16 and, per chunk, generates a question + answer + distractor documents, saved as JSONL for `train_raft.py` to consume.

## Notes

- The router is keyword-based (`router.py`), not learned — extend the keyword lists as you find misclassified queries. Unmatched queries default to Modern / Nationalist.
- `ExpertRetriever` returns a dummy score (`0.0`) after filtering since phase 0's rerank score isn't preserved post-filter — don't read into it, only use it to confirm retrieval happened.
- Rebuild both BM25 and the FAISS index together if `chunks_meta.json` changes — their ordering must stay in sync.
- `raft_data_conversion.ipynb` currently has a Kaggle API key hardcoded in a cell — rotate it and load from an env var / Kaggle secret before pushing or sharing the notebook, same as the earlier Neo4j credential issue in phase 2.