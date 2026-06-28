# Phase 2 — GraphRAG

Phase 2 is now an **extension** of phase 0 and phase 1, not a parallel stack.

The main rule is simple:

- **phase 0** remains the canonical hybrid retriever
- **phase 1** remains the autonomous orchestration layer on top of phase 0
- **phase 2** adds graph + timeline + verification, but it **reuses phase 0 retrieval and shared prompting**

## What this phase does

- Uses **Neo4j** for relationship-heavy retrieval
- Uses **phase 0 hybrid retrieval** for semantic/vector fallback and supplemental evidence
- Uses **SQLite/Pandas** for timeline and chronology questions
- Uses a **router** to decide whether a query is graph, timeline, vector, or verification oriented
- Supports **multi-source verification** when historians disagree
- Exports a **relationship visualization** as PNG

## What changed in this version

- The old standalone lexical vector search was removed from the active path
- The phase 0 retriever is now reused for vector search in phase 2
- The synthesizer now uses the shared prompt builder and phase 0 generation helper
- Hard-coded Neo4j secrets were removed from source
- The CLI now executes the real graph app directly instead of a disconnected reasoning engine

## Folder layout

```text
phase_2_graph_rag/
├── config.py
├── extractor.py
├── graph.py
├── neo4j_store.py
├── router.py
├── run_query.py
├── schema.py
├── state.py
├── synthesizer.py
├── timeline_store.py
├── utils.py
├── vector_search.py
├── visualize.py
├── verification.py
├── tools/
│   └── shared.py
└── README.md
```

## Runtime flow

```text
USER QUERY
    ↓
Intent router
    ↓
┌────────────────────────────────────────────┐
│ graph / timeline / vector / verification   │
└────────────────────────────────────────────┘
    ↓
Evidence merge
    ↓
Shared prompt builder (phase 0)
    ↓
Phase 0 generation helper
    ↓
Final answer
```

## How phase 2 reuses earlier code

**Shared**
- `shared/prompts/rag_prompts.py` builds the final answer prompt
- `shared/vector_store/faiss_store.py` still handles FAISS creation/loading
- `shared/embeddings/embedder.py` still powers embedding generation

**Phase 0**
- `phase_0_rag_baseline/retriever.py` provides the hybrid BM25 + FAISS + reranker stack
- `phase_0_rag_baseline/llm.py` provides the generation helper used by the synthesizer
- `phase_0_rag_baseline/rag_chain.py` shows the canonical prompt+generation pattern

**Phase 1**
- `phase_1_agentic_rag/tools/retrieve_tool.py` already reuses the phase 0 retriever, and phase 2 now follows the same principle

## Setup

Create a Neo4j instance locally and set:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export NEO4J_DATABASE=neo4j
```


## Ingest the corpus

```bash
python -m phase_2_graph_rag.run_query --ingest --reset --page-mode
```

This will:

1. Read the JSONL corpus
2. Extract heuristic nodes and edges
3. Upsert them into Neo4j
4. Populate the timeline SQLite store

## Ask a question

```bash
python -m phase_2_graph_rag.run_query --query "Who fought whom in 1526?"
python -m phase_2_graph_rag.run_query --query "What happened before and after the Kalinga war?"
python -m phase_2_graph_rag.run_query --query "Compare how RC Majumdar and Jadunath Sarkar describe the same event."
```
## retreival mode Stracture
run_query.py
   ↓
app.invoke()
   ↓
GraphRetriever.retrieve()
   ↓
_build_cypher()
   ↓
Neo4j query

### Visualise mode structure
run_query.py
   ↓
export_subgraph_html()
   ↓
store.subgraph()
   ↓
Neo4j


## Routing examples

### 1) Simple fact
Route: **phase 0 hybrid vector search**

Example:
`What is the significance of the Kalinga war?`

### 2) Relationship question
Route: **graph / Cypher**

Example:
`Who fought whom in 1526?`

### 3) Timeline question
Route: **SQL / Pandas timeline store**

Example:
`What happened in the years around 1857?`

### 4) Multi-source verification
Route: **graph + phase 0 vector evidence + timeline**

Example:
`Compare RC Majumdar and Jadunath Sarkar on the same event.`

## Visualization

After ingest:

```bash
python -m phase_2_graph_rag.run_query --visualize "Akbar"
```

This writes a PNG subgraph into:

```text
phase_2_graph_rag/artifacts/visualizations/
```

## Notes

- The extractor is heuristic, so the graph improves as your corpus coverage improves.
- If Neo4j is not available, the code falls back to an in-memory graph so you can still test the routing and synthesis path.
- The synthesizer uses the shared prompt builder and the phase 0 generation helper, so answer style stays aligned across phases.
- The phase 2 vector path is now a bridge into the phase 0 retriever, not a separate implementation.
