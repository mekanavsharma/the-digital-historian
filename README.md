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
| Phase 2   | Planned    | Graph RAG, tool use, multi-hop reasoning |
| Phase 3   | Planned    | MoE & Retrieval Augmented Fine-Tuning |

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

## Repository Structure

```
historyProject/
├── data_pipeline/            # Data extraction & preprocessing scripts
├── index/                    # Generated FAISS + BM25 indexes (gitignored)
├── eval/                     # Evaluation results
├── phase_0_rag_baseline/     # Current working Phase-0 pipeline
├── phase_1_agentic_rag/      # Current working Phase-1 pipeline
├── shared/                   # Reusable components (gradio, embeddings, prompts, etc.)
├── requirements.txt
└── README.md
```
