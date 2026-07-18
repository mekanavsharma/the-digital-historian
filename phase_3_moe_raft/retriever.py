# phase_3_moe_raft/retriever.py

"""
ExpertRetriever – wraps Phase 0's Retriever to filter results by
expert_domain / historian_perspective.
"""
from typing import List, Tuple, Dict
from langchain_core.documents import Document
from phase_0_rag_baseline.retriever import Retriever
from phase_0_rag_baseline.config import RetrievalConfig

class ExpertRetriever:
    def __init__(self, phase0_retriever: Retriever, retrieval_cfg: RetrievalConfig):
        self.retriever = phase0_retriever   # Phase 0 Retriever instance
        self.cfg = retrieval_cfg            # contains k values, weights, etc.

    def retrieve(self, query: str, domain: str, perspective: str,
                 top_k: int = None) -> List[Tuple[float, Document]]:
        """
        Use Phase 0 hybrid retrieval, then keep only documents whose
        metadata matches the given domain & perspective.
        Returns list of (score, Document) – the score is the final rerank score.
        """
        # 1. Get the full set of retrieved & reranked documents from Phase 0
        docs = self.retriever.retrieve(query, self.cfg)   # returns List[Document]
        # 2. Filter by expert metadata
        filtered = []
        for doc in docs:
            meta = doc.metadata
            if (meta.get("expert_domain") == domain and
                meta.get("historian_perspective") == perspective):
                filtered.append(doc)
        # Limit to top_k (default from config or argument)
        if top_k is None:
            top_k = self.cfg.rerank_k
        filtered = filtered[:top_k]
        # Since the Phase 0 retriever may not provide scores after reranking,
        # we return a dummy score of 0.0 (you can adjust if needed)
        return [(0.0, doc) for doc in filtered]