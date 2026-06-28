"""Aggregate JSONL chunks by page, then extract using rule‑based heuristic."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from .extractor import extract_facts_from_record, build_fact_table
from .utils import normalize_text


def get_page_id(chunk_id: str) -> str:
    """Extract page ID from chunk_id (remove trailing _digit)."""
    if not chunk_id:
        return ""
    parts = chunk_id.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return chunk_id


def group_chunks_by_page_id(records: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by page_id extracted from chunk_id."""
    groups = defaultdict(list)
    for rec in records:
        chunk_id = rec.get("chunk_id") or rec.get("metadata", {}).get("chunk_id")
        if not chunk_id:
            continue
        page_id = get_page_id(chunk_id)
        groups[page_id].append(rec)
    return groups


def merge_chunk_content(chunks: List[Dict[str, Any]]) -> str:
    """Merge content of all chunks belonging to the same page."""
    chunks_sorted = sorted(chunks, key=lambda x: x.get("chunk_id", ""))
    texts = []
    for ch in chunks_sorted:
        content = ch.get("content", "")
        # Fix hyphenated line breaks
        content = content.replace("-\n", "").replace("-\r\n", "")
        content = content.replace("\n", " ")
        texts.append(content)
    full = " ".join(texts)
    return normalize_text(full)


def build_page_record(page_id: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a single record for a full page."""
    first = chunks[0]
    merged_content = merge_chunk_content(chunks)
    record = dict(first)
    record["content"] = merged_content
    record["chunk_id"] = page_id
    if "metadata" not in record:
        record["metadata"] = first.get("metadata", {})
    return record


def ingest_pages_rule_based(
    records: Iterable[Dict[str, Any]],
    graph_store,
    timeline_store,
    batch_size: int = 200,
    progress=None,
) -> Dict[str, int]:
    """
    Ingest by grouping chunks into pages, then use rule‑based extraction.
    """
    groups = group_chunks_by_page_id(records)
    totals = {"records": 0, "nodes": 0, "edges": 0}
    batch_nodes = []
    batch_edges = []
    timeline_batch = []  # list of (page_rec, nodes, edges)

    for page_id, chunks in groups.items():
        page_rec = build_page_record(page_id, chunks)
        nodes, edges = extract_facts_from_record(page_rec)

        batch_nodes.extend(nodes)
        batch_edges.extend(edges)
        timeline_batch.append((page_rec, nodes, edges))
        totals["records"] += 1

        if progress:
            progress.update(len(chunks))  # progress over total chunks

        if len(batch_nodes) >= batch_size:
            graph_store.ingest_batch(batch_nodes, batch_edges)
            # Build fact table for timeline store from all pages in this batch
            fact_rows = []
            for pr, n, e in timeline_batch:
                fact_rows.extend(build_fact_table([pr], [{"nodes": n, "edges": e}]))
            if fact_rows:
                timeline_store.ingest_rows(fact_rows)
            totals["nodes"] += len(batch_nodes)
            totals["edges"] += len(batch_edges)
            batch_nodes = []
            batch_edges = []
            timeline_batch = []

    # Final flush
    if batch_nodes:
        graph_store.ingest_batch(batch_nodes, batch_edges)
        fact_rows = []
        for pr, n, e in timeline_batch:
            fact_rows.extend(build_fact_table([pr], [{"nodes": n, "edges": e}]))
        if fact_rows:
            timeline_store.ingest_rows(fact_rows)
        totals["nodes"] += len(batch_nodes)
        totals["edges"] += len(batch_edges)

    return totals