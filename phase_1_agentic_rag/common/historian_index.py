# phase_1_agentic_rag/common/historian_index.py
"""
HistorianIndexStore: Manages FAISS indices for each historian.
- Loads documents and organizes them by historian.
- Provides fuzzy matching for historian names.
- Caches loaded FAISS retrievers for efficiency.
"""

import os
import re
from typing import Dict, List, Any
from collections import defaultdict
from difflib import SequenceMatcher

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from phase_0_rag_baseline.config import RetrievalConfig, ModelConfig
from phase_0_rag_baseline.ingest import load_jsonl_as_documents


class HistorianIndexStore:

    def __init__(self, paths_cfg, model_cfg: ModelConfig, retrieval_cfg: RetrievalConfig):
        self.paths_cfg = paths_cfg
        self.model_cfg = model_cfg
        self.retrieval_cfg = retrieval_cfg

        self.index_cache: Dict[str, Any] = {}
        self.docs_by_historian: Dict[str, List] = defaultdict(list)

        self._prepare_docs()

    def _normalize(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(text).lower())

    def _fuzzy_match(self, query: str, candidates: List[str], threshold=0.75):
        best_match = None
        best_score = 0

        for cand in candidates:
            score = SequenceMatcher(None, query, cand).ratio()
            if score > best_score:
                best_score = score
                best_match = cand

        if best_score >= threshold:
            return best_match

        return None

    def _prepare_docs(self):
        docs = load_jsonl_as_documents(self.paths_cfg.documents_path)

        for doc in docs:
            historian_raw = doc.metadata.get("historian", "unknown")
            historian = self._normalize(historian_raw)

            self.docs_by_historian[historian].append(doc)

        print(f"Prepared {len(self.docs_by_historian)} historian buckets.")
        print(f"Available historians: {list(self.docs_by_historian.keys())[:10]} ...")

    def get_index(self, historian: str):
        norm_query = self._normalize(historian)

        # 1. Exact match
        if norm_query in self.docs_by_historian:
            resolved = norm_query
            print(f"[IndexStore] Exact match for '{historian}' → '{resolved}'")

        else:
            # 2. Fuzzy match
            resolved = self._fuzzy_match(norm_query, list(self.docs_by_historian.keys()))
            if resolved:
                print(f"[IndexStore] Fuzzy match for '{historian}' → '{resolved}'")
            else:
                print(f"[IndexStore] No match for '{historian}'")
                return None

        # 3. Return cached FAISS if exists
        if resolved in self.index_cache:
            return self.index_cache[resolved]

        # 4. LOAD FROM DISK (THIS IS THE FIX)
        index_path = os.path.join(self.paths_cfg.vector_store_path, resolved)

        if not os.path.exists(index_path):
            print(f"[IndexStore]  Missing FAISS folder: {index_path}")
            return None

        print(f"[IndexStore] Loading FAISS from: {index_path}")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.model_cfg.embedding_model,
            model_kwargs={"trust_remote_code": True}
        )

        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": self.retrieval_cfg.rerank_k}
        )

        self.index_cache[resolved] = retriever
        return retriever