# History RAG Project

**A Multi-Phase Retrieval-Augmented Generation (RAG) System for Indian History**

This project builds a specialized, accurate, and grounded Question-Answering system focused on **Indian History**.

The goal is to create a reliable RAG pipeline that answers complex historical questions using OCR-extracted books and documents while minimizing hallucinations through strict grounding and citations.

---


## Project Structure (Phases)

| Phase | Status     | Description |
|-------|------------|-----------|
| **Phase 0** | ✅ Completed | Basic Hybrid RAG (BM25 + FAISS + Reranker + LLM) |
| Phase 1   | Planned    | Query rewriting, Agentic RAG, Chat Memory |
| Phase 2   | Planned    | Graph RAG, tool use, multi-hop reasoning |
| Phase 3   | Planned    | MoE & Retrieval Augmented Fine-Tuning |

---

## The Journey 🚀
I’m currently diving into the world of RAG (Retrieval-Augmented Generation). This project is a 5–6 months sprint broken down into roughly 7 phases.
Since I’m learning these concepts on the go, the specifics of each phase are still a bit fluid. If you have experience with LLMs or vector databases and want to drop some knowledge, I’d love to have you along for the ride!

---

## Current Release: Phase 0

**Phase-0: RAG Baseline** is a clean, modular, and fully local RAG pipeline.

**Key Features:**
- Hybrid retrieval (BM25 + FAISS)
- Cross-encoder reranking
- Qwen2.5-3B-Instruct (4-bit quantized)
- Strict citation grounding
- Configurable answer styles (`short` / `concise` / `detailed`)
- One-time index building (fast subsequent runs)
- Built-in evaluation harness

**Dataset:**
The processed corpus (167 MB) is hosted on Kaggle:
[Input Data (JSONL) - Indian History Corpus](https://www.kaggle.com/datasets/kanav608/input-data)

---

## Quick Start (Phase 0)

```bash
git clone https://github.com/mekanavsharma/the-digital-historian.git
cd the-digital-historian

# Install dependencies
pip install -r requirements.txt

# Build indexes + evaluation (run once)
python -m phase_0_rag_baseline.run_query --evaluate

# Ask questions
python -m phase_0_rag_baseline.run_query --query "Explain the significance of the Rowlatt Act and the Jallianwala Bagh massacre." --answer_style detailed --max_words 500
```

---

## Repository Structure

```
historyProject/
├── phase_0_rag_baseline/     # Current working Phase-0 pipeline
├── shared/                   # Reusable components (embeddings, prompts, etc.)
├── data_pipeline/            # Data extraction & preprocessing scripts
├── index/                    # Generated FAISS + BM25 indexes (gitignored)
├── eval/                     # Evaluation results
├── requirements.txt
└── README.md
```
