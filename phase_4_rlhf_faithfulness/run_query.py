# phase_4_rlhf_faithfulness/run_query.py

import argparse
import sys
import traceback

from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever, Retriever
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig

from phase_3_moe_raft.router import route_query
from phase_3_moe_raft.retriever import ExpertRetriever
from phase_3_moe_raft.raft_model import RAFTModel
from phase_3_moe_raft.config import TOP_K, RAFT_MODEL_PATH

from phase_4_rlhf_faithfulness.self_correction import SelfCorrectingRAFTModel
from phase_4_rlhf_faithfulness.config import DPO_OUTPUT_DIR

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
    bm25 = build_bm25_retriever(docs, bm25_path=paths_cfg.bm25_path)
    faiss_store = build_faiss_retriever(
        docs, model_name=model_cfg.embedding_model,
        index_path=paths_cfg.vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size,
    )
    reranker = load_cross_encoder(model_cfg.reranker_model)
    base_retriever = Retriever(bm25=bm25, faiss=faiss_store, reranker=reranker)

    _COMPONENTS = {"base_retriever": base_retriever, "retrieval_cfg": retrieval_cfg}
    return _COMPONENTS


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Trust (RLHF + Self-Correction)")
    parser.add_argument("--query", required=True)
    parser.add_argument("--domain", choices=["Ancient", "Medieval", "Modern"])
    parser.add_argument("--perspective", choices=["Nationalist", "Marxist", "Neutral"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--model-path", default=DPO_OUTPUT_DIR,
                        help="Defaults to the DPO-tuned model; pass phase_3's RAFT_MODEL_PATH to compare pre-DPO.")
    parser.add_argument("--no-self-correct", action="store_true",
                        help="Skip the critique/refine loop and just show the first draft.")
    parser.add_argument("--verbose", action="store_true", help="Print every round of the self-correction trace.")
    args = parser.parse_args()

    try:
        comp = get_components()
        expert_retriever = ExpertRetriever(comp["base_retriever"], comp["retrieval_cfg"])

        model = RAFTModel(model_path=args.model_path)

        if args.domain:
            domain = args.domain
        else:
            _, expert_dict = route_query(args.query)
            domain = expert_dict["expert_domain"]

        if args.perspective:
            perspective = args.perspective
        else:
            _, expert_dict = route_query(args.query)
            perspective = expert_dict["historian_perspective"]

        docs = expert_retriever.retrieve(args.query, domain, perspective, top_k=args.top_k)
        if not docs:
            print("No matching documents found.")
            return
        doc_texts = [doc.page_content for _, doc in docs]

        print(f"\n=== Domain={domain}, Perspective={perspective}, docs={len(doc_texts)} ===")

        if args.no_self_correct:
            from phase_4_rlhf_faithfulness import prompts
            from phase_4_rlhf_faithfulness.self_correction import _raw_generate
            gen_prompt = prompts.build_generation_prompt(args.query, doc_texts, domain, perspective)
            answer = _raw_generate(model, gen_prompt)
            print(f"\nAnswer (no self-correction):\n{answer}")
        else:
            wrapped = SelfCorrectingRAFTModel(model)
            trace = wrapped.answer(args.query, doc_texts, domain, perspective, verbose=args.verbose)
            print(f"\nFinal answer (after {len(trace.rounds)} round(s), "
                  f"stopped_reason={trace.stopped_reason}, score={trace.final_score:.2f}):\n{trace.final_answer}")

    except Exception:
        print("\n--- UNEXPECTED ERROR ---", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
