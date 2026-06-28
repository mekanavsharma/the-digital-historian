"""Final answer synthesis for Phase 2 GraphRAG.

The synthesizer reuses the shared prompt builder and the phase 0 generation
helper so answer formatting stays consistent across phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from phase_0_rag_baseline.llm import generate_answer
from shared.prompts.rag_prompts import build_prompt

from .utils import normalize_text


@dataclass
class EvidenceDoc:
    page_content: str
    metadata: Dict[str, Any]


def _first_non_empty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, "", [], {}, ()):
            return value
    return None


def _clean_metadata(metadata: Optional[Dict[str, Any]], fallback_chunk_id: str) -> Dict[str, Any]:
    meta = dict(metadata or {})
    if not meta.get("chunk_id"):
        meta["chunk_id"] = fallback_chunk_id
    return meta


def _vector_docs(state: Dict[str, Any]) -> List[EvidenceDoc]:
    vector_results = state.get("vector_results", {}) or {}
    docs = vector_results.get("documents") or []
    rows = vector_results.get("results") or []

    out: List[EvidenceDoc] = []
    if docs:
        for idx, doc in enumerate(docs, start=1):
            metadata = _clean_metadata(getattr(doc, "metadata", {}) or {}, f"vector-{idx}")
            out.append(EvidenceDoc(page_content=getattr(doc, "page_content", "") or "", metadata=metadata))
        return out

    for idx, row in enumerate(rows, start=1):
        metadata = _clean_metadata(row.get("metadata") or {}, row.get("chunk_id") or f"vector-{idx}")
        out.append(EvidenceDoc(page_content=row.get("text", "") or "", metadata=metadata))
    return out


def _graph_docs(state: Dict[str, Any]) -> List[EvidenceDoc]:
    graph_results = state.get("graph_results", {}) or {}
    rows = graph_results.get("rows") or []
    out: List[EvidenceDoc] = []

    for idx, row in enumerate(rows, start=1):
        # Safely extract fields
        source = row.get("source_name")
        relation = row.get("relation")
        target = row.get("target_name")
        if not source or not relation or not target:
            # If essential fields are missing, skip this row
            continue

        props = row.get("properties") or {}
        source_props = row.get("source_properties") or {}
        target_props = row.get("target_properties") or {}

        chunk_id = _first_non_empty(
            props.get("chunk_id"),
            props.get("source_chunk_id"),
            source_props.get("source_chunk_id"),
            target_props.get("source_chunk_id"),
            props.get("page"),
            f"graph-{idx}",
        )

        year = row.get("year") or props.get("year")
        labels = row.get("source_labels") or []
        source_label = labels[0] if labels else row.get("source_label") or "Node"

        page_content = f"Graph evidence: {source_label}:{source} --{relation}--> {target}"
        if year is not None:
            page_content += f" (year={year})"

        context = _first_non_empty(
            props.get("context"),
            target_props.get("context"),
            source_props.get("context")
        )
        if context:
            page_content += f" | text={normalize_text(str(context))[:1200]}"

        if props:
            page_content += f" | properties={props}"

        metadata = dict(source_props)
        metadata.update(target_props)
        metadata.update(props)

        out.append(EvidenceDoc(page_content=page_content, metadata=_clean_metadata(metadata, str(chunk_id))))

    return out


def _timeline_docs(state: Dict[str, Any]) -> List[EvidenceDoc]:
    rows = (state.get("timeline_results", {}) or {}).get("rows") or []
    out: List[EvidenceDoc] = []
    for idx, row in enumerate(rows, start=1):
        chunk_id = _first_non_empty(row.get("chunk_id"), row.get("id"), row.get("source_name"), f"timeline-{idx}")
        page_content = (
            f"Timeline evidence: year={row.get('year')} | "
            f"{row.get('source_label') or 'Node'}:{row.get('source_name')} --{row.get('relation')}--> "
            f"{row.get('target_label') or 'Node'}:{row.get('target_name')}"
        )
        if row.get("text"):
            page_content += f" | text={normalize_text(str(row.get('text')))[:260]}"
        out.append(EvidenceDoc(page_content=page_content, metadata=_clean_metadata(row, str(chunk_id))))
    return out


def _verification_docs(state: Dict[str, Any]) -> List[EvidenceDoc]:
    verdict = state.get("verification_results") or {}
    if not verdict:
        return []
    summary = verdict.get("summary") or "Verification available."
    page_content = f"Verification summary: {summary}. Agreement={verdict.get('agreement')} Overlap={verdict.get('overlap_score')}"
    return [EvidenceDoc(page_content=page_content, metadata={"chunk_id": "verification-summary"})]


def _select_evidence_docs(state: Dict[str, Any]) -> List[EvidenceDoc]:
    intent = state.get("intent", "vector")
    graph_docs = _graph_docs(state)
    vector_docs = _vector_docs(state)
    timeline_docs = _timeline_docs(state)
    verification_docs = _verification_docs(state)

    if intent == "verification":
        return graph_docs + vector_docs + timeline_docs + verification_docs
    if intent == "graph":
        # Always include graph_docs, even if empty, so the fallback can use the raw graph rows
        return graph_docs + (vector_docs[:3] if not graph_docs else vector_docs[:2]) + verification_docs
    if intent == "timeline":
        return timeline_docs + (vector_docs[:3] if not timeline_docs else vector_docs[:2]) + verification_docs
    return vector_docs + graph_docs[:2] + timeline_docs[:2] + verification_docs


def _deterministic_summary(state: Dict[str, Any], used_chunk_ids: Sequence[str], max_words: int) -> str:
    parts: List[str] = []
    query = state.get("original_query", "")
    intent = state.get("intent", "vector")

    if query:
        parts.append(f"Query: {query}")
    parts.append(f"Intent: {intent}")

    graph_rows = (state.get("graph_results", {}) or {}).get("rows") or []
    if graph_rows:
        top_graph = []
        for row in graph_rows[:5]:
            top_graph.append(f"{row.get('source_name')} --{row.get('relation')}--> {row.get('target_name')}")
        parts.append("Graph evidence: " + "; ".join(top_graph))

    timeline_rows = (state.get("timeline_results", {}) or {}).get("rows") or []
    if timeline_rows:
        top_timeline = []
        for row in timeline_rows[:3]:
            top_timeline.append(f"{row.get('year')}: {row.get('source_name')} {row.get('relation')} {row.get('target_name')}")
        parts.append("Timeline evidence: " + "; ".join(top_timeline))

    vector_rows = (state.get("vector_results", {}) or {}).get("results") or []
    if vector_rows:
        snippets = []
        for row in vector_rows[:3]:
            snippet = normalize_text(str(row.get("text", "")))[:180]
            snippets.append(f"[{row.get('chunk_id')}] {snippet}")
        parts.append("Text evidence: " + " | ".join(snippets))

    verification = state.get("verification_results") or {}
    if verification:
        parts.append(f"Verification: {verification.get('summary', '')}")

    if used_chunk_ids:
        parts.append("Citations: " + " ".join(f"[chunk_id={cid}]" for cid in dict.fromkeys(used_chunk_ids) if cid))

    answer = "\n".join(parts).strip()
    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]) + " ..."
    return answer


def synthesize_answer(state: Dict[str, Any], llm_tuple=None, answer_style: str = "concise", max_words: int = 300) -> str:
    query = state.get("original_query", "")
    evidence_docs = _select_evidence_docs(state)

    # ---- DEBUG: print raw graph rows ----
    graph_rows_raw = (state.get("graph_results", {}) or {}).get("rows") or []
    print(f"[DEBUG] Raw graph rows count: {len(graph_rows_raw)}")
    if graph_rows_raw:
        print("[DEBUG] First 4 rows:")
        for row in graph_rows_raw[:4]:
            print(f"  {row.get('source_name')} --{row.get('relation')}--> {row.get('target_name')}")

    # If no evidence docs, construct a summary from raw graph rows
    if not evidence_docs:
        # Try to extract any available evidence from graph results
        if graph_rows_raw:
            # Build a simple summary from graph rows
            summaries = []
            for row in graph_rows_raw[:5]:
                src = row.get("source_name", "unknown")
                rel = row.get("relation", "RELATED_TO")
                tgt = row.get("target_name", "unknown")
                summaries.append(f"{src} --{rel}--> {tgt}")
            if summaries:
                return f"Graph evidence found: " + "; ".join(summaries) + ". Please refine your query for a more detailed answer."
            else:
                return "I don't know."
        if state.get("retrieval_mode") == "graph":
            return (
                "I could not find an answer in the Neo4j graph for this query. "
                "That usually means the relevant entity or relationship was not extracted into the graph during ingestion."
            )
        fallback = _deterministic_summary(state, [], max_words=max_words)
        return fallback or "I don't know."

    # Build prompt and generate answer
    prompt, used_chunk_ids = build_prompt(
        evidence_docs,
        query,
        answer_style=answer_style,
        max_words=max_words,
    )

    if llm_tuple is None:
        return _deterministic_summary(state, used_chunk_ids, max_words=max_words)

    answer = generate_answer(
        prompt,
        llm_tuple,
        chunk_ids_used=used_chunk_ids,
        answer_style=answer_style,
        max_words=max_words,
    )
    return answer