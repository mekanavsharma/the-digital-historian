# shared/evaluation/metrics.py
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from typing import List
from langchain_core.documents import Document
from rouge_score import rouge_scorer

from shared.vector_store.faiss_store import faiss_as_retriever
from phase_0_rag_baseline.reranker import rerank_with_cross_encoder
from shared.prompts.rag_prompts import build_prompt
from phase_0_rag_baseline.llm import generate_answer

def rouge_scores(predicted: str, reference: str):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    sc = scorer.score(reference, predicted)
    return {"rouge1": sc["rouge1"].fmeasure, "rougeL": sc["rougeL"].fmeasure}


def evaluate_generation(
    eval_csv_path: str,
    retriever_fn,
    reranker_model=None,
    hf_tuple=None,
    k=7
    ):

    """
    eval_csv must have columns: question,correct_answer
    retriever_fn: function(question, k) -> list of candidate Documents
    reranker_model: CrossEncoder or None. If provided, rerank top candidates before generating.
    hf_tuple: (tokenizer, model) or None (mock)
    Returns: pandas Series with average rouge1 and rougeL
    """

    df = pd.read_csv(eval_csv_path)
    rr1 = []
    rrL = []
    for _, row in df.iterrows():
        q = str(row["question"])
        ref = str(row["correct_answer"])

        # 1) Retrieve candidates
        candidates = retriever_fn(q, k)  # expects a function that returns list of docs

        # 2) Rerank with cross-encoder if provided
        if reranker_model is not None:
            final_docs = rerank_with_cross_encoder(reranker_model, q, candidates, top_n=k)
        else:
            final_docs = candidates[:k]

        # 3) Build prompt
        prompt, _ = build_prompt(final_docs, q)

        # 4) Generate
        pred = generate_answer(prompt, hf_tuple)

        # 5) Score ROUGE
        sc = rouge_scores(pred, ref)
        rr1.append(sc["rouge1"])
        rrL.append(sc["rougeL"])

    return pd.Series({"rouge1": float(np.mean(rr1)), "rougeL": float(np.mean(rrL))})


def _hybrid_merge_dedupe(
        bm25_docs: List[Document],
        faiss_docs: List[Document],
        k: int,
    ):
        """
        Weighted fusion by reciprocal rank (simple and interpretable).
        BM25 and FAISS produce ranked lists; we assign score = weight * (1/(rank+k)).
        Then we merge and return top-k documents by combined score.
        """
        bm25_weight = 0.4
        faiss_weight = 0.6

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


def evaluate_all_retrievers(
    eval_csv_path: str,
    bm25,
    faiss,
    reranker_model=None,
    hf_tuple=None,
    k=10
):
    """
    Evaluate only the HYBRID retriever (BM25 + FAISS merged + Weighted scoring)
    Return a single-row DataFrame with rows per retriever and rouge metrics.

    Args:
        eval_csv_path: Path to evaluation CSV with columns 'question' and 'correct_answer'
        bm25: BM25Retriever object
        faiss: FAISS vectorstore object
        reranker_model: CrossEncoder for reranking (optional)
        hf_tuple: (tokenizer, model) for generation
        k: Number of documents to retrieve for evaluation
    """
    results = []

    # Wrap retriever functions to accept (q, kk)
    def bm25_fn(q, kk=k):
        """BM25 retriever with proper error handling."""
        try:
            bm25.k = kk
            return bm25.invoke(q)
        except Exception:
            return bm25.get_relevant_documents(q)

    def faiss_fn(q, kk=k):
        """FAISS retriever with proper error handling."""
        try:
            faiss_retriever = faiss_as_retriever(faiss, k=kk)
            return faiss_retriever.invoke(q)
        except Exception:
            return faiss_retriever.get_relevant_documents(q)

    def hybrid_fn(q, kk=k):
        """Hybrid retrieval combining BM25 and FAISS."""
        bm25_docs = bm25_fn(q, kk)
        faiss_docs = faiss_fn(q, kk)
        merged = _hybrid_merge_dedupe(bm25_docs, faiss_docs, k=kk)
        if not merged:
            print(f" Warning: No documents retrieved for query: {q[:50]}...")
        return merged

    print("Evaluating generation with retriever: HYBRID")
    try:
        sc = evaluate_generation(
            eval_csv_path,
            hybrid_fn,
            reranker_model=reranker_model,
            hf_tuple=hf_tuple,
            k=k
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        # Return empty results instead of crashing
        return pd.DataFrame([{"retriever": "HYBRID", "rouge1": 0.0, "rougeL": 0.0}])

    return pd.DataFrame([{"retriever": "HYBRID", "rouge1": sc["rouge1"], "rougeL": sc["rougeL"]}])