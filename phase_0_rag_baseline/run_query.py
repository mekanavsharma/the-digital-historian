# phase_0_rag_baseline/run_query.py
# This is the single entrypoint your call
import argparse
import os
import pandas as pd
import traceback

from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever, Retriever
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.llm import load_llm
from phase_0_rag_baseline.rag_chain import RAGChain
from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig
from shared.evaluation.metrics import evaluate_all_retrievers

# max_new_tokens define how many words will be generated

"""
Lifecycle:
1. One-time (cached in memory + disk where applicable):
   - Load documents
   - Build BM25 index (saved to disk)
   - Build FAISS index (saved to disk)
   - Load reranker
   - Load LLM

2. Per-query:
   - Retrieval (BM25 + FAISS hybrid)
   - Reranking
   - Prompting + generation
"""


# Global caches (one-time build)
_PIPELINE = None
_COMPONENTS = None   # docs, bm25, faiss, reranker, hf_tuple
_EVAL_RESULTS = None

def build_components(
    retrieval_cfg: RetrievalConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
):
    """Build (or reuse) all heavy components once."""

    global _COMPONENTS
    if _COMPONENTS is not None:
        print("Components already built and cached. Reusing.")
        return _COMPONENTS

    print("\n=== Building components (one-time) ===")

    # -------------------------
    # Load documents
    # -------------------------
    docs = load_jsonl_as_documents(paths_cfg.documents_path)
    print(f"Loaded {len(docs)} documents.")

    # -------------------------
    # Build retrievers
    # -------------------------
    bm25 = build_bm25_retriever(
        docs,
        bm25_path=paths_cfg.bm25_path,
    )
    print("BM25 ready.")

    faiss = build_faiss_retriever(
        docs,
        model_name=model_cfg.embedding_model,
        index_path=paths_cfg.vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size,
    )
    print("FAISS ready.")

    # -------------------------
    # Reranker + LLM
    # -------------------------
    reranker = load_cross_encoder(model_cfg.reranker_model)
    print("Reranker loaded.")

    hf_tuple = load_llm(model_cfg.llm_model)
    print("HF LLM loaded.")

    _COMPONENTS = {
        "docs": docs,
        "bm25": bm25,
        "faiss": faiss,
        "reranker": reranker,
        "hf_tuple": hf_tuple,
    }

    print("=== All components ready ===\n")
    return _COMPONENTS


def build_pipeline(
    retrieval_cfg: RetrievalConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
):

    """Build (or reuse) the full RAG pipeline."""

    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # Build/reuse shared components first
    comp = build_components(retrieval_cfg, model_cfg, paths_cfg)

    # -------------------------
    # Assemble pipeline
    # -------------------------
    retriever = Retriever(
        bm25=comp["bm25"],
        faiss=comp["faiss"],
        reranker=comp["reranker"],
    )

    rag = RAGChain(
        retriever=retriever,
        hf_tuple=comp["hf_tuple"],
    )

    _PIPELINE = rag
    return rag

def run_evaluation_once(
    retrieval_cfg: RetrievalConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
):
    """Run evaluation using the already-built components (no duplication)."""

    global _EVAL_RESULTS
    if _EVAL_RESULTS is not None:
        print("Evaluation already computed. Skipping.")
        return _EVAL_RESULTS

    if os.path.exists(paths_cfg.eval_result_path):
        print("Loading cached evaluation results from disk.")
        df = pd.read_csv(paths_cfg.eval_result_path)
        _EVAL_RESULTS = df
        return df

    print("\nRunning ONE-TIME evaluation...\n")

    # Reuse components built by build_pipeline()
    comp = build_components(retrieval_cfg, model_cfg, paths_cfg)

    try:
        df = evaluate_all_retrievers(
            eval_csv_path=paths_cfg.eval_csv_path,
            bm25=comp["bm25"],
            faiss=comp["faiss"],
            reranker_model=comp["reranker"],
            hf_tuple=comp["hf_tuple"],
            k=retrieval_cfg.hybrid_k,
        )
    except FileNotFoundError as e:
        print(f"Evaluation failed: {e}")
        print(f"Make sure eval_csv_path exists: {paths_cfg.eval_csv_path}")
        raise
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        traceback.print_exc()
        raise

    # Save results
    os.makedirs(os.path.dirname(paths_cfg.eval_result_path), exist_ok=True)
    df.to_csv(paths_cfg.eval_result_path, index=False)

    _EVAL_RESULTS = df
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase-0  RAG query")

    # --query and --evaluate are mutually exclusive, one of them must be supplied
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="User query (required in normal mode)")
    group.add_argument("--evaluate", action="store_true", help="Run one-time evaluation and exit")

    parser.add_argument("--answer_style", type=str, choices=["short", "concise", "detailed"], default="concise", help="Answer style: short | concise | detailed (default: concise)")
    parser.add_argument("--max_words", type=int, default=None, help="Maximum word limit for answer (optional)")

    args = parser.parse_args()

    # -------------------------
    # Load configs
    # -------------------------
    retrieval_cfg = RetrievalConfig()
    model_cfg = ModelConfig()
    paths_cfg = PathsConfig()

    # -------------------------
    # ONE-TIME EVALUATION MODE
    # -------------------------
    if args.evaluate:
        # we don't need the full RAG pipeline for evaluation; components are built inside
        df = run_evaluation_once(
            retrieval_cfg=retrieval_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg
        )
        print("\n=== Evaluation Results ===\n")
        print(df)
        exit(0)

    # -------------------------
    # Build pipeline (only needed for queries)
    # -------------------------
    rag = build_pipeline(
        retrieval_cfg=retrieval_cfg,
        model_cfg=model_cfg,
        paths_cfg=paths_cfg
    )

    # -------------------------
    # NORMAL QUERY MODE
    # -------------------------
    # At this point args.query is guaranteed (evaluate branch exited above)
    print("\nQuery:", args.query)
    answer = rag.answer(
        query=args.query,
        cfg=retrieval_cfg,
        answer_style=args.answer_style,
        max_words=args.max_words
    )

    print("\n=== Answer ===\n")
    print(answer)

