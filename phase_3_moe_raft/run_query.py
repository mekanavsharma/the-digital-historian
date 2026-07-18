# phase_3_moe_raft/run_query.py

import argparse, sys, traceback
from pathlib import Path

# Phase 0 imports
from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever, Retriever
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig

from phase_3_moe_raft.router import route_query
from phase_3_moe_raft.retriever import ExpertRetriever
from phase_3_moe_raft.raft_model import RAFTModel
from phase_3_moe_raft.config import TOP_K, RAFT_MODEL_PATH, BASE_LLM_MODEL_PATH

# Global cache for Phase 0 components
_COMPONENTS = None

def get_components():
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS

    retrieval_cfg = RetrievalConfig()
    retrieval_cfg.rerank_k = 25
    model_cfg = ModelConfig()
    paths_cfg = PathsConfig()

    docs = load_jsonl_as_documents(paths_cfg.documents_path)
    print(f"Loaded {len(docs)} documents.")

    bm25 = build_bm25_retriever(docs, bm25_path=paths_cfg.bm25_path)
    faiss_store = build_faiss_retriever(
        docs,
        model_name=model_cfg.embedding_model,
        index_path=paths_cfg.vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size
    )
    reranker = load_cross_encoder(model_cfg.reranker_model)

    # Build the Phase 0 Retriever (hybrid + reranker)
    base_retriever = Retriever(bm25=bm25, faiss=faiss_store, reranker=reranker)

    _COMPONENTS = {
        "base_retriever": base_retriever,
        "retrieval_cfg": retrieval_cfg,
    }
    return _COMPONENTS

def main():
    parser = argparse.ArgumentParser(description="Phase 3: MoE + RAFT Query")
    parser.add_argument("--query", required=True)
    parser.add_argument("--domain", choices=["Ancient","Medieval","Modern"])
    parser.add_argument("--perspective", choices=["Nationalist","Marxist","Neutral"])
    parser.add_argument("--show-docs", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--raft-model", action="store_true",
                    help="Use the fine‑tuned RAFT model instead of the base LLM")
    parser.add_argument("--model-path", default=None, help="Override any model path (ignores --raft-model if given)")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    try:
        comp = get_components()
        expert_retriever = ExpertRetriever(comp["base_retriever"], comp["retrieval_cfg"])
        print("Expert retriever ready.")

        model = None
        if not args.no_model:
            try:
                # Decide which model to load
                if args.model_path:
                    model_path = args.model_path
                elif args.raft_model:
                    model_path = RAFT_MODEL_PATH
                else:
                    model_path = BASE_LLM_MODEL_PATH   # defined in config
                print(f"Loading model: {model_path} ...")
                model = RAFTModel(model_path=model_path)
                print("Model loaded.")
            except Exception as e:
                print(f"ERROR loading model: {e}", file=sys.stderr)
                print("Falling back to document-only mode.", file=sys.stderr)

        # Determine domain/perspective
        if args.domain:
            domain = args.domain
        else:
            _, expert_dict = route_query(args.query)
            domain = expert_dict["expert_domain"]

        if args.compare:
            perspectives = ["Nationalist", "Marxist", "Neutral"]
        else:
            perspectives = [args.perspective] if args.perspective else [None]

        for persp in perspectives:
            if persp is None:
                _, expert_dict = route_query(args.query)
                persp = expert_dict["historian_perspective"]
                print(f"\n=== Auto-detected: domain={domain}, perspective={persp} ===")
            else:
                print(f"\n=== Domain={domain}, Perspective={persp} ===")

            docs = expert_retriever.retrieve(args.query, domain, persp, top_k=args.top_k)

            if not docs:
                print("No matching documents found.")
                continue

            if args.show_docs or args.no_model:
                print("\nRetrieved documents:")
                for score, doc in docs:
                    meta = doc.metadata
                    print(f"  [{meta.get('historian','?')} | {meta.get('expert_domain','?')}/{meta.get('historian_perspective','?')}] "
                          f"Score: {score:.4f}\n    {doc.page_content[:200]}...")

                # for _, doc in docs:          # ignore the score
                #     meta = doc.metadata
                #     print(f"  [{meta.get('historian','?')} | {meta.get('expert_domain','?')}/{meta.get('historian_perspective','?')}] "
                #         f"\n    {doc.page_content[:200]}...")

            if model and not args.no_model:
                # parts = [f"### Document [{i+1}]: {doc.page_content[:1500]}" for i, (_, doc) in enumerate(docs)]
                parts = [f"### Document [{i+1}]: {doc.page_content[:800]}" for i, (_, doc) in enumerate(docs)]
                docs_str = "\n".join(parts)
                answer = model.answer(docs_str, args.query, domain=domain, perspective=persp)
                print(f"\nAnswer: {answer}")

    except Exception as e:
        print("\n--- UNEXPECTED ERROR ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
