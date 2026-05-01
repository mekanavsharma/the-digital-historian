"""Phase 1 autonomous agent entrypoint.

This keeps the same high-level lifecycle as phase 0:
1. Build components once
2. Reuse them for each query
3. Provide a CLI for direct execution
"""


import argparse
import time
from typing import Dict, List, Optional
import sys
from shared.deploy.gradio_ui import launch_historian_ui

from sentence_transformers import SentenceTransformer

from phase_0_rag_baseline.ingest import load_jsonl_as_documents
from phase_0_rag_baseline.llm import generate_answer, load_llm
from phase_0_rag_baseline.reranker import load_cross_encoder
from phase_0_rag_baseline.retriever import build_bm25_retriever, build_faiss_retriever
from phase_0_rag_baseline.retriever import Retriever

from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig, PathsConfig
from phase_1_agentic_rag.common.graph import Phase1GraphBuilder
from phase_1_agentic_rag.common.state import HistorianState
from phase_1_agentic_rag.tools import (
    ClaimAlignerTool,
    EvaluatorTool,
    FinalSynthesizerTool,
    MemoryManagerTool,
    PlannerTool,
    PositionExtractorTool,
    RetrievalTool,
)
from phase_1_agentic_rag.common.historian_index import HistorianIndexStore

_COMPONENTS_CACHE = {}
_PIPELINE_CACHE = {}

def _hist_key(historians):
    if not historians:
        return ("__all__",)
    return tuple(sorted(normalize_name(h) for h in historians))

import re

def normalize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

from difflib import SequenceMatcher

def is_fuzzy_match(name1, name2, threshold=0.75):
    # Quick check for exact match first (efficiency)
    if name1 == name2:
        return True
    return SequenceMatcher(None, name1, name2).ratio() >= threshold


def build_components(
    retrieval_cfg: RetrievalConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    historians: Optional[List[str]] = None,
):
    """Build or reuse all heavy components once."""
    global _COMPONENTS_CACHE

    cache_key = _hist_key(historians)
    if cache_key in _COMPONENTS_CACHE:
        print("Components already built for this historian set. Reusing.")
        return _COMPONENTS_CACHE[cache_key]

    print("\n=== Building components (one-time) ===")

    docs = load_jsonl_as_documents(paths_cfg.documents_path)
    print(f"Loaded {len(docs)} documents.")

    # -------------------------
    # Build retrievers
    # -------------------------
    bm25 = build_bm25_retriever(docs, bm25_path=paths_cfg.bm25_path)
    print("BM25 ready.")

    faiss = build_faiss_retriever(
        docs,
        model_name=model_cfg.embedding_model,
        index_path=paths_cfg.vector_store_path,
        vectorstore_batch_size=retrieval_cfg.vectorstore_batch_size,
    )
    print("FAISS ready.")

    # ---------------------------------------------------------------------------
    # Get historian-specific FAISS indices ready in the index store
    # ---------------------------------------------------------------------------
    index_store = HistorianIndexStore(paths_cfg, model_cfg, retrieval_cfg)

    # -------------------------
    # Reranker + LLM
    # -------------------------
    reranker = load_cross_encoder(model_cfg.reranker_model)
    print("Reranker loaded.")

    try:
        hf_tuple = load_llm(model_cfg.llm_model)
        print("HF LLM loaded.")
    except Exception as e:
        print("Failed to load HF model automatically (likely resource constraints).")
        print("Error:", e)
        print("Falling back to mock generation (hf_tuple=None).")
        hf_tuple = None

    print("Loading tiny model for routing/entity tasks...")
    small_hf_tuple = load_llm(model_cfg.small_llm_model)
    print("Tiny HF LLM loaded.")

    print("Loading sentence-transformer embedder...")
    embed_model = SentenceTransformer(model_cfg.embedding_model, trust_remote_code=True)

    _COMPONENTS_CACHE[cache_key] = {
        "docs": docs,
        "bm25": bm25,
        "faiss": faiss,
        "reranker": reranker,
        "hf_tuple": hf_tuple,
        "small_hf_tuple": small_hf_tuple,
        "embed_model": embed_model,
        "index_store": index_store,
    }

    print("=== All components ready ===\n")
    return _COMPONENTS_CACHE[cache_key]


def build_pipeline(
    retrieval_cfg: RetrievalConfig,
    model_cfg: ModelConfig,
    paths_cfg: PathsConfig,
    historians: Optional[List[str]] = None,
):
    global _PIPELINE_CACHE

    cache_key = _hist_key(historians)
    if cache_key in _PIPELINE_CACHE:
        print("Pipeline already built for this historian set. Reusing.")
        return _PIPELINE_CACHE[cache_key]

    comp = build_components(retrieval_cfg, model_cfg, paths_cfg, historians=historians)

    # Create full Retriever object (exactly like Phase 0 + your notebook) ===
    full_retriever = Retriever(
        bm25=comp["bm25"],
        faiss=comp["faiss"],
        reranker=comp["reranker"],
    )

    memory_tool = MemoryManagerTool(comp["embed_model"])
    planner_tool = PlannerTool(comp["small_hf_tuple"], generate_answer)
    retrieval_tool = RetrievalTool(          # ← correct
        retriever=full_retriever,            # ← full object here
        retrieval_cfg=retrieval_cfg,
        index_store=comp["index_store"],      # ← pass the index store for historian-specific retrieval
    )
    position_tool = PositionExtractorTool()
    claim_aligner_tool = ClaimAlignerTool(comp["small_hf_tuple"], generate_answer)
    evaluator_tool = EvaluatorTool()
    synthesizer_tool = FinalSynthesizerTool(comp["hf_tuple"], comp["small_hf_tuple"], generate_answer)

    graph = Phase1GraphBuilder(
        memory_tool=memory_tool,
        planner_tool=planner_tool,
        retrieval_tool=retrieval_tool,
        position_tool=position_tool,
        claim_aligner_tool=claim_aligner_tool,
        evaluator_tool=evaluator_tool,
        synthesizer_tool=synthesizer_tool,
    ).build()

    _PIPELINE_CACHE[cache_key] = graph
    return _PIPELINE_CACHE[cache_key]

def run_query(
    question: str,
    chat_memory: Optional[List[Dict[str, str]]] = None,
    answer_style: str = "detailed",
    max_words: Optional[int] = 350,
    historians: Optional[List[str]] = None,
    retrieval_cfg: Optional[RetrievalConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    paths_cfg: Optional[PathsConfig] = None,
    session_id: int = 0,
):
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    model_cfg = model_cfg or ModelConfig()
    paths_cfg = paths_cfg or PathsConfig()

    graph = build_pipeline(retrieval_cfg, model_cfg, paths_cfg, historians=historians)
    state = HistorianState(
        original_query=question,
        chat_history=chat_memory or [],
        answer_style=answer_style,
        max_words=max_words,
        session_id=session_id,
        historian_override=historians,
    )
    return graph.invoke(state)


def run_demo_sequence(
    retrieval_cfg: Optional[RetrievalConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    paths_cfg: Optional[PathsConfig] = None,
    historians: Optional[List[str]] = None,
):
    retrieval_cfg = retrieval_cfg or RetrievalConfig()
    model_cfg = model_cfg or ModelConfig()
    paths_cfg = paths_cfg or PathsConfig()

    samples = [
        ("Tell me what did Guru Gobind Singh brought to sikhi which was not present earlier. What he did that changed the shape of Sikhi", "detailed", 350),
        ("How did he shape it according to political conditions?", "short", 200),
        ("Let's discuss further aspect on militarisation of Sikhi how did he do and how did his devoution to Shakti (Durga or Chandi or Bhagauti)) Goddess helped him", "detailed", 500),
        ("Tell me how he met Banda Singh Bahadur & how Banda Bahadur worked on militarisation of Sikhi after his death", "detailed", 500),
        ("What were the battles did Banda Singh Bahadur fought. Who and how he was captured and killed?", "detailed", 500),
        ("How was Banda singh Bahadur was captured and killed?", "concise", 300),
        ("What was his son's name? Where was he from originally?", "concise", 350),
        ("Can you check and tell when did Ashoka or Asoka became Buddhist before or after Kalinga , why did he became?", "concise", 300),
        ("How did he took on the empire that his grandfather and what were things he added in the empire and made it such a great empire", "detailed", 500),
        ("Can you tell me what were the key things Ahilyabai Holkar did?", "detailed", 500),
        ("What are the key temples she restored and how she did it like Somnath and Kashi?", "detailed", 500),
    ]

    chat_memory: List[Dict[str, str]] = []
    print("=== Phase 1 Autonomous Agent Ready ===\n")

    for idx, (question, answer_style, max_words) in enumerate(samples, start=1):
        start = time.time()
        result = run_query(
            question=question,
            chat_memory=chat_memory,
            answer_style=answer_style,
            max_words=max_words,
            historians=historians,  # Convert to list
            retrieval_cfg=retrieval_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
            session_id=1,
        )

        print(f"\nQuestion: {question}")
        print("Rewritten:", result.get("rewritten_query"))
        print("Final Answer:\n", result.get("final_answer", "No answer"))

        chat_memory = result["chat_history"]
        end = time.time()
        print("Total time taken:", (end - start) / 60)

    return chat_memory


def run_evaluation_once(
    retrieval_cfg: Optional[RetrievalConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    paths_cfg: Optional[PathsConfig] = None,
    historians: Optional[List[str]] = None,
):
    """Smoke test that reuses the notebook's own sample questions."""
    return run_demo_sequence(retrieval_cfg, model_cfg, paths_cfg, historians)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 LangGraph Agent entrypoint")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", type=str, help="Run a single query and exit")
    group.add_argument("--interactive", action="store_true", help="Start interactive chat mode")
    group.add_argument("--evaluate", action="store_true", help="Run demo sequence")

    parser.add_argument("--answer_style", type=str, choices=["short", "concise", "detailed"], default="concise")
    parser.add_argument("--max_words", type=int, default=None)
    parser.add_argument("--historians", type=str, default=None, help="Force a historian label instead of extracting it from the query")

    return parser.parse_args()


def parse_historians(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    historians = [h.strip() for h in raw.split(",")]
    historians = [h for h in historians if h]
    return historians or None

def main():
    args = parse_args()

    # -------------------------
    # Load configs
    # -------------------------
    retrieval_cfg = RetrievalConfig()
    model_cfg = ModelConfig()
    paths_cfg = PathsConfig()
    historians = parse_historians(args.historians)

    if args.evaluate:
        run_evaluation_once(retrieval_cfg, model_cfg, paths_cfg, historians)
        return

    # ====================== INTERACTIVE CHAT MODE ======================
    if args.interactive or not args.query:
        chat_memory: List[Dict[str, str]] = []
        print("=== Phase 1 Digital Historian (Interactive Chat) ===\n")
        print("Type your question and press Enter. Type 'exit', 'quit', or 'q' to stop.\n")

        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            if not question:
                continue

            start = time.time()
            result = run_query(
                question=question,
                chat_memory=chat_memory,
                answer_style=args.answer_style,
                max_words=args.max_words,
                historians=historians,
                retrieval_cfg=retrieval_cfg,
                model_cfg=model_cfg,
                paths_cfg=paths_cfg,
            )

            print(f"Is follow-up: {len(chat_memory) > 0}")
            print(f"\nQuestion: {question}")
            print("Rewritten:", result.get("rewritten_query"))
            print("Final Answer:\n", result.get("final_answer", "No answer"))
            print("Chat history length:", len(result.get("chat_history", [])))
            print("Time Taken:", round((time.time() - start) / 60, 2), "min\n")

            chat_memory = result.get("chat_history", chat_memory)

    # ====================== SINGLE QUERY MODE ======================
    else:  # --query was provided
        start = time.time()
        result = run_query(
            question=args.query,
            answer_style=args.answer_style,
            max_words=args.max_words,
            historians=historians,
            retrieval_cfg=retrieval_cfg,
            model_cfg=model_cfg,
            paths_cfg=paths_cfg,
        )
        print(f"\nQuestion: {args.query}")
        print("Rewritten:", result.get("rewritten_query"))
        print("Final Answer:\n", result.get("final_answer", "No answer"))
        print("Chat history length:", len(result.get("chat_history", [])))
        print("Time Taken:", round((time.time() - start) / 60, 2), "min")




# if __name__ == "__main__":
#     main()

# At the very end of phase_1/run_query.py
if __name__ == "__main__":

    if "--gradio" in sys.argv:
        launch_historian_ui(
            run_query_func=run_query,
            title="Phase 1 - Digital Historian (Agentic)",
            description="Multi-lane historical agent with memory. Citations are shown below each answer.",
        )
    else:
        main()