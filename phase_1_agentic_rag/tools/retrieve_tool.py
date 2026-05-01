# phase_1/tools/retrieve_context_tool.py

"""
Lane-aware retrieval module for LangGraph-based RAG pipeline.
- Supports multi-historian queries within a single lane
- Per-historian retrieval with result aggregation

- route(): dispatches parallel retrieval jobs using LangGraph Send API (one per lane).
- run_lane(): executes retrieval for a single lane, supporting multiple historians per lane.
- retrieve_context(): core retrieval logic:
    • routes to historian-specific FAISS index when available
    • falls back to global hybrid retriever (BM25 + vector + rerank)

Output is structured per lane_id for downstream synthesis.
"""

from typing import Any, Dict, Optional

from phase_0_rag_baseline.retriever import Retriever
from phase_0_rag_baseline.config import RetrievalConfig


class RetrievalTool:
    """
    Reuses the exact hybrid search + reranking logic from phase_0_rag_baseline.
    Adds optional historian-specific filtering (your "lane" logic).
    """
    def __init__(self, retriever: Retriever, retrieval_cfg: RetrievalConfig, index_store=None):
        self.retriever = retriever
        self.retrieval_cfg = retrieval_cfg
        self.index_store = index_store

    def retrieve_context(self, query: str, lane: Optional[Dict] = None) -> Dict[str, Any]:
        """
        lane example: {"historian": "R.C. Majumdar"} or {"historian": "general"}
        If historian != "general", we filter the results after hybrid + rerank.
        """
        historian = None
        if lane:
            historian = lane.get("historian")

        #  If specific historian → use dedicated FAISS
        if historian and historian != "general" and self.index_store:

            faiss = self.index_store.get_index(historian)

            if faiss:
                # docs = faiss.similarity_search(query, k=self.retrieval_cfg.rerank_k)
                docs = faiss.invoke(query)

                return {
                    "query": query,
                    "retrieved_docs": [d.page_content for d in docs],
                    "metadata": [d.metadata for d in docs],
                }

        # fallback to global retriever
        candidates = self.retriever.retrieve(query, self.retrieval_cfg)

        return {
            "query": query,
            "retrieved_docs": [doc.page_content for doc in candidates],
            "metadata": [doc.metadata for doc in candidates]
        }

    def run_lane(self, lane: Dict) -> Dict:
        historians = lane.get("historian", "general")

        MAX_PER_HISTORIAN = 7

        # Handle multiple historians
        if isinstance(historians, str):
            historians = [h.strip() for h in historians.split(",")]

        all_results = []

        # KEY: retrieve separately per historian
        for historian in historians:

            ctx = self.retrieve_context(
                query=lane["question"],
                lane={"historian": historian},
            )

            # Attach historian tag for traceability
            for m in ctx["metadata"]:
                m["__lane_historian"] = historian

            docs = ctx["retrieved_docs"][:MAX_PER_HISTORIAN]
            meta = ctx["metadata"][:MAX_PER_HISTORIAN]

            all_results.extend(list(zip(docs, meta)))

        # Optional: deduplicate
        seen = set()
        final_docs = []
        final_meta = []

        for doc, meta in all_results:
            if doc not in seen:
                seen.add(doc)
                final_docs.append(doc)
                final_meta.append(meta)

        return {
            "retrieved_results": {
                lane["lane_id"]: {
                    "query": lane["question"],
                    "retrieved_docs": final_docs,
                    "metadata": final_meta,
                }
            }
        }

    def route(self, state):
        from langgraph.types import Send

        return [Send("retrieve_tool", lane) for lane in state.lanes]