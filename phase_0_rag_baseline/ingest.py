# phase_0_rag_baseline/ingest.py
import json
import os
from typing import List
from langchain_core.documents import Document
import re

def normalize_ocr_text(text: str) -> str:
    # Fix hyphenated line breaks first (from second function)
    text = re.sub(r"-\n\s*", "", text)
    # Remove page numbers standing alone
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    # Replace single newlines inside paragraphs with space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple newlines to max two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse excessive spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_jsonl_as_documents(jsonl_paths) -> List[Document]:
    """
    Read a single file or a folder of .jsonl files and turn them into Document objects.
    Expects JSONL lines with at least keys: content, chunk_id, volume (with volume_title), chapter_title, page
    """
    if isinstance(jsonl_paths, str):
        if os.path.isdir(jsonl_paths):
            files = [os.path.join(jsonl_paths, f) for f in os.listdir(jsonl_paths) if f.endswith(".jsonl")]
        else:
            files = [jsonl_paths]
    else:
        files = list(jsonl_paths)

    documents = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cleaned_content = normalize_ocr_text(row["content"])  # Apply cleaning here
                documents.append(
                    Document(
                        page_content=cleaned_content,
                        metadata={
                            "chunk_id": row.get("chunk_id"),
                            "volume": row.get("volume", {}).get("volume_title"),
                            "chapter": row.get("chapter_title"),
                            "page": row.get("page"),
                            "historian": row.get("historian"),
                        },
                    )
                )
    return documents
