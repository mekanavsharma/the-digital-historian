# History RAG Project

**A Multi-Phase Retrieval-Augmented Generation (RAG) System for Indian History**

This project builds a specialized, accurate, and grounded Question-Answering system focused on **Indian History**.

The goal is to create a reliable RAG pipeline that answers complex historical questions using OCR-extracted books and documents while minimizing hallucinations through strict grounding and citations.

---


## Project Structure (Phases)

| Phase | Status     | Description |
|-------|------------|-----------|
| **Phase 0**  | ✅ Completed   | Basic Hybrid RAG (BM25 + FAISS + Reranker + LLM) |
| **Phase 1**  | ✅ Completed   | Agentic RAG, Query rewriting,  Chat Memory |
| **Phase 2**   | ✅ Completed   | Graph RAG, tool use, multi-hop reasoning |
| **Phase 3**   | ✅ Completed     | MoE & Retrieval Augmented Fine-Tuning |
| **Phase 4**   | Planned    | RLHF & Self-Correction Loops |

---

## The Journey 🚀
I’m currently diving into the world of RAG (Retrieval-Augmented Generation). This project is a 5–6 months sprint broken down into roughly 7 phases.
Since I’m learning these concepts on the go, the specifics of each phase are still a bit fluid. If you have experience with LLMs or vector databases and want to drop some knowledge, I’d love to have you along for the ride!

---
## 🧠 Phase 0 Highlights (Baseline)
**Phase-0: RAG Baseline** is a clean, modular, and fully local RAG pipeline.

**Key Features:**
- Hybrid retrieval (BM25 + FAISS)
- Cross-encoder reranking
- Strict citation grounding with Configurable answer styles
- Built-in evaluation harness
---

---
## 🧠 Phase 1 Highlights (Agentic RAG)

- LangGraph-based orchestration
- Multi-lane (historian-aware) retrieval
- Claim extraction & comparison
- Iterative evaluation (replan vs finalize)

---

---

## Features
- **Hybrid Retrieval**: BM25 + FAISS with reciprocal rank fusion
- **Cross-encoder reranking**
- **Historian-specific indices** (per-author FAISS)
- **Agentic workflow** using LangGraph (Phase 1)
- **Conversation memory** with follow-up detection
- **Strict grounding** with chunk-level citations `[chunk_id=...]`
- **Configurable answer styles**: `short` / `concise` / `detailed`
- **Gradio UI** for interactive chatting
- **Evaluation harness** with ROUGE metrics

---

---
## 🧠 Phase 2 Highlights (Graph RAG)

- Graph‑based retrieval
- Intent router
- Timeline store
- Iterative evaluation (replan vs finalize)

---

---

## Features
- **Multi‑source verification:** – cross‑checks graph, vector, and timeline evidence to reconcile historian disagreements
- **Rule‑based extractor** – spaCy‑driven entity & relation extraction with strict filtering (period‑agnostic, works for ancient/medieval/modern history)
- **Visualisation** – interactive HTML subgraph export (PyVis) for exploring relationships
- **Reuses Phase 0/1 components** – hybrid retriever (BM25 + FAISS + reranker), shared prompt builder, and generation helper
- **CLI support** – ingest, query, visualise, and force retrieval modes `--retrieval-mode`

---

---
## 🧠 Phase 3 Highlights (MoE & RAFT)

- Keyword-based router (domain + perspective)
- Expert-filtered retrieval
- QLoRA fine-tuned RAFT model
- Base vs RAFT model switch via CLI flag

---

---

## Features
- **Multi‑source verification:** – cross‑checks graph, vector, and timeline evidence to reconcile historian disagreements
- **Rule‑based extractor** – spaCy‑driven entity & relation extraction with strict filtering (period‑agnostic, works for ancient/medieval/modern history)
- **Visualisation** – interactive HTML subgraph export (PyVis) for exploring relationships
- **Reuses Phase 0/1 components** – hybrid retriever (BM25 + FAISS + reranker), shared prompt builder, and generation helper
- **CLI support** – ingest, query, visualise, and force retrieval modes `--retrieval-mode`


- **Query router** – classifies each query into a time domain (Ancient/Medieval/Modern) and a historian perspective (Nationalist/Marxist/Neutral)
- **Expert-filtered retrieval** – wraps the phase 0 hybrid retriever and keeps only chunks matching the routed expert
- **RAFT fine-tuning** – QLoRA adapter (Qwen2.5-3B-Instruct) trained to answer only from retrieved documents, ignoring distractors
- **Dual model support** – --raft-model / --model-path flags switch between the base LLM and the fine-tuned adapter without touching config
- **Comparison mode** – --compare runs a query across all 3 historian perspectives side by side

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/mekanavsharma/the-digital-historian.git
cd the-digital-historian
pip install -r requirements.txt
```

### 2. Download the Data
The processed corpus (167 MB) is hosted on Kaggle:
[Input Data (JSONL) - Indian History Corpus](https://www.kaggle.com/datasets/kanav608/input-data)


### 3. Run the System
**Phase 0 – Baseline RAG**
```bash
python -m phase_0_rag_baseline.run_query --query "When did Ashoka become a Buddhist?" --answer_style concise
```

**Phase 1 – Agentic RAG**
```bash
# Single query
python -m phase_1_agentic_rag.run_query --query "How did Ashoka rule his subjects?" --historians "Romila Thapar,R.K. Mukherjee"

# Interactive chat
python -m phase_1_agentic_rag.run_query --interactive

# Launch Gradio UI
python -m phase_1_agentic_rag.run_query --gradio
```

**Phase 2 – Graph RAG**
```bash
# Single Query
python -m phase_2_graph_rag.run_query --query "Who were associated with Abhinava bharat"

# Force Usage of Graph RAG
python -m phase_2_graph_rag.run_query --query "Who were associated with Abhinava bharat" --retrieval-mode graph

# Get Graph Visualisation
python -m phase_2_graph_rag.run_query --visualize "Akbar"
```

**Phase 2 – MoE RAFT**
```bash
# Base model (no fine-tuning), compare all 3 perspectives, show retrieved docs
python -m phase_3_moe_raft.run_query --query "What was the significance of Gandhi's Non-Cooperation Movement?" --compare --show-docs

# RAFT fine-tuned model
python -m phase_3_moe_raft.run_query --query "Why was Bengal partitioned?" --compare --raft-model

# Fine Tune model
python -m phase_3_moe_raft.train_raft
```

## Repository Structure

```bash
historyProject/
├── data_pipeline/            # Data extraction & preprocessing scripts
├── index/                    # Generated FAISS + BM25 indexes (gitignored)
├── eval/                     # Evaluation results
├── phase_0_rag_baseline/     # Current working Phase-0 pipeline
├── phase_1_agentic_rag/      # Current working Phase-1 pipeline
├── phase_2_graph_rag/        # Current working Phase-2 pipeline
├── phase_3_moe_raft/         # Current working Phase-3 pipeline
├── shared/                   # Reusable components (gradio, embeddings, prompts, etc.)
├── requirements.txt
└── README.md
```
