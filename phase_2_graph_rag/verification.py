"""Multi-source verification helper."""

from __future__ import annotations

from typing import Any, Dict, List, Set


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def compare_sources(graph_results: List[Dict[str, Any]], vector_results: List[Dict[str, Any]], timeline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    graph_entities: Set[str] = set()
    for row in graph_results or []:
        graph_entities.add(_normalize(row.get("source_name")))
        graph_entities.add(_normalize(row.get("target_name")))

    vector_hits: Set[str] = set()
    for row in vector_results or []:
        meta = row.get("metadata") or {}
        vector_hits.add(_normalize(meta.get("historian")))
        vector_hits.add(_normalize(meta.get("chunk_id") or row.get("chunk_id")))
        text = _normalize(row.get("text"))
        for entity in graph_entities:
            if entity and entity in text:
                vector_hits.add(entity)

    timeline_years = {str(row.get("year")) for row in timeline_results or [] if row.get("year") is not None}
    timeline_entities: Set[str] = set()
    for row in timeline_results or []:
        timeline_entities.add(_normalize(row.get("source_name")))
        timeline_entities.add(_normalize(row.get("target_name")))

    overlap = len((graph_entities & vector_hits) | (graph_entities & timeline_entities))
    agreement = overlap > 0 or bool(timeline_years)
    summary = (
        "Graph, text, and timeline evidence overlap on at least one entity or year."
        if agreement
        else "No strong overlap detected; answer cautiously and prefer explicit evidence."
    )

    return {
        "graph_entity_count": len({e for e in graph_entities if e}),
        "vector_hit_count": len({e for e in vector_hits if e}),
        "timeline_year_count": len(timeline_years),
        "overlap_score": overlap,
        "agreement": agreement,
        "summary": summary,
    }
