# phase_0_rag_baseline/reranker.py

"""
Provides functions to load a SentenceTransformers CrossEncoder and
to rerank a list of documents given a query.
"""
from typing import List
try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS = True
except Exception:
    CrossEncoder = None
    _HAS_CROSS = False

def load_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
    if not _HAS_CROSS:
        raise ImportError(
            "sentence_transformers.CrossEncoder not available. "
            "pip install sentence-transformers"
        )
    return CrossEncoder(model_name)

def rerank_with_cross_encoder(
    reranker,
    query: str,
    documents: List,
    top_n: int,
):
    """
    documents: list of Document objects (candidates)
    returns: reranked list of Document objects (top_n)
    """
    if reranker is None:
        return documents[:top_n]

    MAX_RERANK_CHARS = 1024
    passages = [d.page_content[:MAX_RERANK_CHARS] for d in documents]
    pairs = [(query, p) for p in passages]
    scores = reranker.predict(pairs, batch_size=32)  # higher is better
    scored = list(zip(scores, documents))
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored[:top_n]]
    return reranked
