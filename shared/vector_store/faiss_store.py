# shared/vector_store.py
"""
Vector store wrapper for FAISS using langchain_community.vectorstores.FAISS.
This is shared because vector stores are reused across phases.
"""
import os
from typing import List
from langchain_core.documents import Document

import faiss
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore


try:
    from langchain_community.vectorstores import FAISS as _FAISS
except Exception:
    _FAISS = None


def build_faiss_vectorstore(
    documents: List[Document],
    embeddings,
    index_path: str,
    batch_size: int = 512,
    ):
    """
    Build or load a FAISS vectorstore.
    - If index exists at index_path → load (NO embedding)
    - Else → embed documents, build index, save to index_path
    - Supports batch processing + tqdm progress bar.
    - Handles resuming from partial index if build was interrupted
    """
    if _FAISS is None:
        raise RuntimeError("FAISS vectorstore (langchain_community.vectorstores.FAISS) not available.")

    # -------------------------
    # LOAD EXISTING INDEX
    # -------------------------
    os.makedirs(index_path, exist_ok=True)
    index_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")
    vectorstore = None
    if os.path.exists(index_file) and os.path.exists(pkl_file):
        vectorstore = _FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        current_num = vectorstore.index.ntotal
        print(f"Loaded existing FAISS index from {index_path} with {current_num} documents.")
    else:
        print("No existing index files found.")

    # -------------------------
    # BUILD NEW INDEX OR RESUME FROM PARTIAL
    # -------------------------
    if vectorstore is None or vectorstore.index.ntotal < len(documents):
        if not documents:
            raise ValueError("No documents provided.")

        if vectorstore is None:
            # Initialize new empty FAISS index
            print("Initializing new FAISS index...")
            dummy_embedding = embeddings.embed_query("test")
            dim = len(dummy_embedding)
            vectorstore = _FAISS(
                embedding_function=embeddings.embed_query,
                index=faiss.IndexFlatL2(dim),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            print(f"Created empty FAISS index (dim={dim})")
            resume_from = 0
        else:
            # Resume from partial
            resume_from = vectorstore.index.ntotal
            print(f"Resuming build from {resume_from} documents (out of {len(documents)} total).")

        # Batch-wise addition with progress bar
        remaining_docs = len(documents) - resume_from
        num_batches = (remaining_docs + batch_size - 1) // batch_size
        print(f"Adding remaining {remaining_docs} documents in {num_batches} batches...")

        for i in tqdm(range(resume_from, len(documents), batch_size), desc="Building FAISS", total=num_batches):
            batch = documents[i:i + batch_size]
            vectorstore.add_documents(batch)

        # -------------------------
        # SAVE INDEX
        # -------------------------
        vectorstore.save_local(index_path)
        print(f"FAISS index successfully saved to → {index_path} with {vectorstore.index.ntotal} documents.")
    else:
        print("Index already complete; no additional documents to add.")

    return vectorstore

# Small helper to return retriever
def faiss_as_retriever(vectorstore, k: int = 10):
    return vectorstore.as_retriever(search_kwargs={"k": k})
