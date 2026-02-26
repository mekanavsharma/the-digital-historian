# shared/embeddings/embeddings.py
"""
Embedding wrapper that tries to use the LangChain/HuggingFace wrappers you used in the notebook.
This file lives in shared/ because embeddings are reusable across phases.
"""

import torch

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs: dict = None
    ):
        if HuggingFaceEmbeddings is None:
            raise RuntimeError(
                "HuggingFaceEmbeddings not available. Install langchain-huggingface."
            )

        self.model_name = model_name
        model_kwargs = {"trust_remote_code": True, "device": "cuda" if torch.cuda.is_available() else "cpu"}
        self.impl = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    def embed_documents(self, texts):
        return self.impl.embed_documents(texts)

    def embed_query(self, query):
        return self.impl.embed_query(query)
