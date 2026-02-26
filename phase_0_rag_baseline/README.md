# Phase-0 RAG baseline - History QA

A clean, modular Retrieval-Augmented Generation (RAG) pipeline for India-specific history questions from OCR-extracted JSONL corpus.

Built with:
- **Hybrid retrieval** (BM25 + FAISS)
- **Cross-encoder reranking**
- **Qwen2.5-3B-Instruct** (quantized 4-bit)
- **Strict citation grounding**

## Features
- One-time index building (FAISS + BM25) → instant reload
- Hybrid search + reranker
- Configurable answer styles (`short` / `concise` / `detailed`)
- Automatic chunk citations
- Evaluation harness (retrieval + generation metrics)
- Fully local, no external APIs

## Dataset
The processed corpus (`data_jsonl/`) is **not included** in the GitHub repository because of its size (167 MB).

**Download the full dataset here:**
[Input Data (JSONL)](https://www.kaggle.com/datasets/kanav608/input-data)

### How to use:
1. Download the dataset from the link above
2. Extract and place all `.jsonl` files inside `data_pipeline/data_jsonl/`
3. Run the pipeline — it will automatically load them.

## Entrypoint:

### Clone & Install
```bash
git clone https://github.com/mekanavsharma/the-digital-historian.git
cd the-digital-historian
pip install -r requirements.txt
```

### One-time evaluation
One-time full evaluation (saved to eval/eval_results.csv)
- `python -m phase_0_rag_baseline.run_query.py --evaluate`

### Normal user query
- `python -m phase_0_rag_baseline.run_query.py --query "your question" --answer_style detailed --max_words 300`

### Sample query result

```bash
python -m phase_0_rag_baseline.run_query --query "Can you check and tell when did Ashoka or Asoka became Buddhist before or after Kalinga , why did he became?" --answer_style concise --max_words 250
```

```
Query: Can you check and tell when did Ashoka or Asoka became Buddhist before or after Kalinga , why did he became?

=== Answer ===

Ashoka became Buddhist before the Kalinga war. According to the context, he became a Buddhist before the Kalinga war, around 265 BCE, as an upasaka. The remorse he felt after the Kalinga war deepened his belief in Buddhism, leading him to adopt it as a central pursuit. This conversion was not immediate but developed gradually over time, culminating in a strong commitment to Buddhist principles and practices. The Kalinga war served as a catalyst for his spiritual transformation, prompting him to seek solace and guidance in Buddhism.

[chunk_id=HCIP_RCM_V2_C5_P143_1] [chunk_id=ASOKA_RKM_C2_P43_1] [chunk_id=EARLY_INDIA_RTH_C6_P288_0] [chunk_id=GEM_LOTUS_AE_C6_P486_1] [chunk_id=AL_HIND_AW_V2_C10_P176_2] [chunk_id=EARLY_INDIA_RTH_C6_P262_1] [chunk_id=WTWI_ALB_C3_P112_1]
```


### Evaluation On current config

```bash
python -m phase_0_rag_baseline.run_query --evaluate
```

```
=== All components ready ===

Evaluating generation with retriever: HYBRID

=== Evaluation Results ===

  retriever    rouge1    rougeL
0    HYBRID  0.442593  0.287597
```


## Project Structure

```
/historyProject/
├── index/
│   ├── faiss/
│   └── bm25.pkl
├── data_pipeline/
│   ├── data_csv/
│   ├── data_jsonl/
│   ├── data_extraction.ipynb
│   └── data_preparation.py
├── phase_0_rag_baseline/
│   ├── retriever.py
│   ├── reranker.py
│   ├── rag_chain.py
│   ├── run_query.py
│   ├── llm.py
│   ├── config.py
│   └── ingest.py
├── shared/
│   ├── embeddings/embeddings.py
│   ├── evaluation/metrics.py
│   ├── prompts/rag_prompts.py
│   └── vector_store/vector_store.py
├── eval/
├── .gitignore
└── README.md
```

**Notes:**
- Update `config.py` to fit your requriments.
- The data corpus is huge on Kaggle P100 GPU it took 35 minutes to build the index. After that we are just reading it. No need to re-create again.
- The LLM & reranker loads will fail if dependencies or resources (GPU/RAM) are not present; in that case the pipeline will falls.
