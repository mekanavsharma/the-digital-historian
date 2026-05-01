# Digital Historian

A two-phase Retrieval-Augmented Generation system for historical Q&A,
built with LangChain, FAISS, BM25, and LangGraph.

---

## Project Structure

```
digital_historian/
├──phase_1_agentic_rag/
│    │
│    ├── common/                     # Shared utilities & configuration
│    │   ├── __init__.py
│    │   ├── graph.py               # nodes and edges defined for agents
│    │   ├── historian_index.py     # Get FAISS index for specific historian
│    │   ├── state.py               # Define LangGraph State
│    │   └── utils.py               # OCR text normalisation, followup detection function
│    │
│    ├── tools/                      # All LangGraph nodes (one concern per file)
│    │   ├── __init__.py
│    │   ├── claim_aligner.py        # Multi-historian comparison (optional)
│    │   ├── evaluator.py            # Evidence sufficiency check + router
│    │   ├── final_synthesizer.py   # Grounded answer + memory update
│    │   ├── memory_manager.py       # Follow-up detection + history management
│    │   ├── planner.py              # Query decomposition → lanes
│    │   ├── position_extractor.py   # Raw evidence → structured positions
│    │   ├── query_rewriter.py       # Pronoun resolution via small LLM
│    │   └── retriever_tool.py            # retrieve_context, retrieve_tool, retrieve_node
│    │
│    ├── __init__.py
│    ├── README.md
│    └── run_query.py                     # CLI entry point (--phase 1)
│
├── shared/
│   ├── deploy/gradio_ui.py
│   ├── embeddings/embeddings.py
│   ├── evaluation/metrics.py
│   ├── prompts/rag_prompts.py
│   └── vector_store/vector_store.py
├── eval/
├── .gitignore
└── README.md
```

```
## Layout

- `common/` — shared state, graph wiring, and utility helpers
- `tools/` — one class per notebook node
- `run_query.py` — phase-0-style entrypoint with build-once / reuse-later flow
- `config.py` — model, retrieval, and path defaults
```
---

## Architecture

### Phase 0 – Baseline RAG

```
User Question
     ↓
Hybrid Search  (BM25 + FAISS, Reciprocal Rank Fusion)
     ↓
Cross-Encoder Rerank
     ↓
build_prompt
     ↓
LLM → Answer
```

### Phase 1 – Autonomous Agent (LangGraph)

```
User Question
     ↓
memory_manager   ← detects follow-up, manages chat history
     ↓
planner          ← rewrites query, decomposes into lanes
     ↓  (Send API – parallel fan-out, one per lane)
retrieve_tool    ← hybrid search → historian filter → rerank
     ↓
position_extractor  ← structures evidence per lane
     ↓
claim_aligner    ← optional multi-historian comparison
     ↓
evaluator        ← is evidence sufficient?
     ↓                       ↑
     └─── needs_replan ──────┘
     ↓  evidence OK
final_synthesizer  ← grounded answer + memory update
     ↓
END
```

---

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

### Configure paths

Edit `common/config.py` to point `DATA_FOLDER`, `FAISS_INDEX_PATH`, and
`BM25_PATH` at your actual data.

### Phase 0 – one question

```bash
python -m phase_0_rag_baseline.run_query --query "When did Ashoka become a Buddhist?" --answer_style concise
```

### Phase 1 – interactive multi-turn session

```bash
python -m phase_1_newtesting.run_query --interactive
```

### Phase 1 - in Gradio UI
```bash
python -m phase_1_newtesting.run_query --gradio
```

### Phase 1 - Historian comparison
```bash
python -m phase_1_newtesting.run_query --query "How did Asoka rule upon his subjects?" --answer_style concise --max_words 250 --historian "Romila Thapar, RK Mukherjee"
```


### Python API

```python
# Phase 0
from phase_0.run_query import build_pipeline, run_query

pipeline = build_pipeline()
answer = run_query("Who was Banda Singh Bahadur?", pipeline, answer_style="concise")

# Phase 1
from phase_1.agent import build_agent, run_agent

agent, chat_memory = build_agent()
result, chat_memory = run_agent("Tell me about Guru Gobind Singh", agent, chat_memory)

# Follow-up (pronouns resolved automatically)
result, chat_memory = run_agent("How did he shape Sikhi militarily?", agent, chat_memory)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| `common/` package | Single source of truth for config, avoiding drift between phases |
| One file per node in `tools/` | Easy to unit-test, swap, or extend individual nodes |
| Factory functions (`make_retrieve_tool`, etc.) | Avoids globals; heavy objects are closed over cleanly |
| `partial()` binding in `agent.py` | Keeps node signatures LangGraph-compatible (single-arg) |
| `merge_dicts` reducer on `retrieved_results` / `positions` | Allows parallel Send lanes to write independently then merge |
| Small LLM for routing/rewriting | Keeps latency low for planning; large LLM reserved for synthesis |