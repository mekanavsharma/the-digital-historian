"""Configuration for phase 2 GraphRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from phase_0_rag_baseline.config import ModelConfig, PathsConfig


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


def _default_artifacts_dir() -> str:
    env_dir = os.getenv("GRAPH_RAG_ARTIFACTS_DIR")
    if env_dir:
        return env_dir
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return str(kaggle_working / "phase_2_graph_rag" / "artifacts")
    return "phase_2_graph_rag/artifacts"


@dataclass
class GraphRAGConfig:
    # Core paths
    documents_path: str = os.getenv("GRAPH_RAG_DOCUMENTS_PATH", "data_pipeline/data_jsonl")
    artifacts_dir: str = os.getenv("GRAPH_RAG_ARTIFACTS_DIR", _default_artifacts_dir())
    bm25_path: str = os.getenv("GRAPH_RAG_BM25_PATH", PathsConfig.bm25_path)
    vector_store_path: str = os.getenv("GRAPH_RAG_VECTOR_STORE_PATH", PathsConfig.vector_store_path)

    # Neo4j connection (environment only; no secrets in source)
    neo4j_uri: str = os.getenv("NEO4J_URI", "neo4j+s://XCA.databases.neo4j.io")
    neo4j_user: str = os.getenv("NEO4J_USER", "XCA")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "QRWR")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "XCA")

    # Retrieval / routing
    vector_top_k: int = _env_int("GRAPH_RAG_VECTOR_TOP_K", 5)
    graph_top_k: int = _env_int("GRAPH_RAG_GRAPH_TOP_K", 20)
    timeline_top_k: int = _env_int("GRAPH_RAG_TIMELINE_TOP_K", 20)
    vectorstore_batch_size: int = _env_int("GRAPH_RAG_VECTORSTORE_BATCH_SIZE", 512)

    # Shared phase-0 style retrieval settings
    embedding_model: str = os.getenv("GRAPH_RAG_EMBEDDING_MODEL", ModelConfig.embedding_model)
    reranker_model: str = os.getenv("GRAPH_RAG_RERANKER_MODEL", ModelConfig.reranker_model)

    # Extraction mode
    extraction_mode: str = os.getenv("GRAPH_RAG_EXTRACTION_MODE", "heuristic")

    # Optional synthesis model hooks
    llm_model: str = os.getenv("GRAPH_RAG_LLM_MODEL", ModelConfig.llm_model)
    small_llm_model: str = os.getenv("GRAPH_RAG_SMALL_LLM_MODEL", ModelConfig.small_llm_model)
    enable_synthesis_llm: bool = os.getenv("GRAPH_RAG_ENABLE_SYNTHESIS_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}

    # Behaviour
    ingest_limit: int | None = None
    reset_graph_on_ingest: bool = False
    fallback_to_memory_graph: bool = True
    fallback_to_simple_search: bool = True

    def ensure_dirs(self) -> None:
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_path(self) -> str:
        return str(Path(self.artifacts_dir) / "timeline.sqlite")

    @property
    def graph_export_dir(self) -> str:
        return str(Path(self.artifacts_dir) / "visualizations")
