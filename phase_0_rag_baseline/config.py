# phase_0_rag_baseline/config.py

from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    bm25_k: int = 50
    faiss_k: int = 60
    hybrid_k: int = 30
    rerank_k: int = 10
    max_turns: int = 5
    vectorstore_batch_size: int = 512
    bm25_weight: float = 0.4
    faiss_weight: float = 0.6


@dataclass
class ModelConfig:
    embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    small_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class PathsConfig:
    vector_store_path: str = "index/faiss/"
    bm25_path: str = "index/bm25.pkl"
    documents_path: str ="data_pipeline/data_jsonl"
    eval_csv_path: str = "eval/retrieval_eval.csv"
    eval_result_path: str = "eval/eval_results.csv"
