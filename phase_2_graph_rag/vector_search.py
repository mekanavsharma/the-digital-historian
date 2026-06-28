"""Vector retrieval bridge for phase 2.

This module reuses the phase 0 hybrid retrieval stack instead of maintaining
a second standalone lexical retriever.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from phase_0_rag_baseline.config import RetrievalConfig
from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import Retriever, build_bm25_retriever, build_faiss_retriever
from phase_0_rag_baseline.reranker import load_cross_encoder


@lru_cache(maxsize=4)
def _load_retriever(
    documents_path: str,
    bm25_path: str,
    vector_store_path: str,
    embedding_model: str,
    reranker_model: str,
    vectorstore_batch_size: int,
) -> Retriever:
    docs = load_jsonl_as_documents(documents_path)
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)

    retrieval_cfg = RetrievalConfig(vectorstore_batch_size=vectorstore_batch_size)
    bm25 = build_bm25_retriever(docs, bm25_path=bm25_path)
    faiss = build_faiss_retriever(
        docs,
        model_name=embedding_model,
        index_path=vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size,
    )
    reranker = load_cross_encoder(reranker_model)

    return Retriever(bm25=bm25, faiss=faiss, reranker=reranker)


class VectorSearchTool:
    def __init__(self, cfg, top_k: Optional[int] = None):
        self.cfg = cfg
        self.top_k = top_k or cfg.vector_top_k
        self.retrieval_cfg = RetrievalConfig()

    @property
    def retriever(self) -> Retriever:
        return _load_retriever(
            documents_path=self.cfg.documents_path,
            bm25_path=self.cfg.bm25_path,
            vector_store_path=self.cfg.vector_store_path,
            embedding_model=self.cfg.embedding_model,
            reranker_model=self.cfg.reranker_model,
            vectorstore_batch_size=self.cfg.vectorstore_batch_size,
        )

    def _to_result_rows(self, docs: List[Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for rank, doc in enumerate(docs[: self.top_k], start=1):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            text = getattr(doc, "page_content", "") or ""
            chunk_id = metadata.get("chunk_id") or metadata.get("id") or f"vector-{rank}"
            rows.append(
                {
                    "rank": rank,
                    "score": round(1.0 / rank, 4),
                    "chunk_id": str(chunk_id),
                    "text": text,
                    "metadata": metadata,
                }
            )
        return rows

    def search(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        limit = k or self.top_k
        docs = self.retriever.retrieve(query, self.retrieval_cfg)
        rows = self._to_result_rows(docs[:limit])
        return {
            "query": query,
            "results": rows,
            "documents": docs[:limit],
            "retrieved_docs": [r["text"] for r in rows],
            "metadata": [r["metadata"] for r in rows],
            "chunk_ids": [r["chunk_id"] for r in rows],
        }

    def retrieve(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        return self.search(query, k=k)
