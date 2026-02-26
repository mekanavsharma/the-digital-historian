# phase_0_rag_baseline/retriever.py

"""
Phase-0 Retriever.

Responsibilities:
- Build BM25 and FAISS retrievers using shared wrappers
- Perform hybrid retrieval (BM25 + FAISS)
- Deduplicate by chunk_id
- Optionally rerank using cross-encoder

All K values are controlled via cfg:
- cfg.bm25_k
- cfg.faiss_k
- cfg.hybrid_k
- cfg.rerank_k
"""

import os
import pickle
from typing import List
from langchain_core.documents import Document

try:
    from langchain_community.retrievers import BM25Retriever
except Exception:
    BM25Retriever = None

from shared.embeddings.embedder import EmbeddingModel
from shared.vector_store.faiss_store import build_faiss_vectorstore, faiss_as_retriever
from phase_0_rag_baseline.reranker import rerank_with_cross_encoder


# -------------------------
# Builder functions
# -------------------------

def build_bm25_retriever(documents: List[Document], bm25_path: str = "bm25.pkl"):
    """
    documents: list of Document objects
    returns: bm25 retriever (LangChain wrapper)
    """
    """
    Build or load BM25 retriever.
    If bm25_path exists, load it; else build and save.
    """
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        print(f"Loaded BM25 retriever from {bm25_path}")
    else:
        if BM25Retriever is None:
            raise RuntimeError("langchain_community.retrievers.BM25Retriever not available in environment.")
        bm25 = BM25Retriever.from_documents(documents)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        print(f"Built and saved BM25 retriever to {bm25_path}")
    return bm25


def build_faiss_retriever(
    documents: List[Document],
    model_name: str,
    index_path: str,
    vectorstore_batch_size: int,
    ):
    """
    Build a FAISS retriever via LangChain community wrapper.
    """
    embeddings = EmbeddingModel(
        model_name=model_name,
        encode_kwargs={"batch_size": 128}
    ).impl

    vectorstore = build_faiss_vectorstore(
        documents=documents,
        embeddings=embeddings,
        index_path=index_path,
        batch_size=vectorstore_batch_size,
    )
    return vectorstore


# -------------------------
# Retriever class
# -------------------------

class Retriever:
    def __init__(self, bm25=None, faiss=None, reranker=None):
        self.bm25 = bm25
        self.faiss = faiss
        self.reranker = reranker

    def _hybrid_merge_dedupe(
        self,
        bm25_docs: List[Document],
        faiss_docs: List[Document],
        k: int,
        bm25_weight: float,
        faiss_weight: float,
    ):
        """
        Weighted fusion by reciprocal rank (simple and interpretable).
        BM25 and FAISS produce ranked lists; we assign score = weight * (1/(rank+k)).
        Then we merge and return top-k documents by combined score.
        """
        scores = {}          # mapping of chunk_id(cid) -> score
        doc_map = {}         # mapping of chunk_id(cid) -> document

        for rank, doc in enumerate(bm25_docs):
            cid = doc.metadata["chunk_id"] if not isinstance(doc, dict) else doc["metadata"]["chunk_id"]
            if cid not in doc_map: # Keep only 1 reference
                doc_map[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + bm25_weight * (1.0 / (rank + k))

        for rank, doc in enumerate(faiss_docs):
            cid = doc.metadata["chunk_id"] if not isinstance(doc, dict) else doc["metadata"]["chunk_id"]
            doc_map[cid] = doc
            scores[cid] = scores.get(cid, 0.0) + faiss_weight * (1.0 / (rank + k))


        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True) # x[1] will ensure we sort by 'value' not by 'key'
        out = [doc_map[cid] for cid, _ in ranked[:k]]   # ranked has chunk_id-score pair, we want to get docs as per there chunk_id
        return out

    def retrieve(self, query: str, cfg):
        """
        cfg must expose:
        - bm25_k
        - faiss_k
        - hybrid_k
        - rerank_k
        """
        bm25_docs = []
        faiss_docs = []

        # BM25 retrieval
        if self.bm25 is not None:
            try:
                self.bm25.k = cfg.bm25_k
                bm25_docs = self.bm25.invoke(query)
            except Exception:
                try:
                    bm25_docs = self.bm25.get_relevant_documents(query)
                except Exception:
                    bm25_docs = []

        # FAISS retrieval
        if self.faiss is not None:
            faiss_retriever = faiss_as_retriever(self.faiss, k=cfg.faiss_k)
            try:
                faiss_docs = faiss_retriever.invoke(query)
            except Exception:
                try:
                    faiss_docs = faiss_retriever.get_relevant_documents(query)
                except Exception:
                    faiss_docs = []

        # Hybrid search/merge
        merged = self._hybrid_merge_dedupe(bm25_docs, faiss_docs, k=cfg.hybrid_k, bm25_weight=cfg.bm25_weight, faiss_weight=cfg.faiss_weight)

        # Optional reranking
        if self.reranker is not None:
            merged = rerank_with_cross_encoder(self.reranker, query, merged, top_n=cfg.rerank_k)

        return merged
