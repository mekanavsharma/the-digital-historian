"""LangGraph-style orchestration for phase 2 GraphRAG."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .config import GraphRAGConfig
from .extractor import build_fact_table, extract_facts_from_records
from .neo4j_store import GraphStore, QuerySpec
from .router import IntentRouter, infer_entities_from_query
from .schema import RELATION_TYPES
from .state import GraphRAGStateDict
from .synthesizer import synthesize_answer
from .timeline_store import TimelineStore
from .vector_search import VectorSearchTool
from .verification import compare_sources

try:
    from langgraph.graph import END, StateGraph  # type: ignore
    _HAS_LANGGRAPH = True
except Exception:
    END = "__END__"
    StateGraph = None
    _HAS_LANGGRAPH = False


class GraphRAGTools:
    def __init__(self, cfg: GraphRAGConfig, graph_store: GraphStore):
        self.cfg = cfg
        self.graph_store = graph_store
        self.router = IntentRouter()
        self.vector_tool = VectorSearchTool(cfg, top_k=cfg.vector_top_k)
        self.timeline_store = TimelineStore(cfg.sqlite_path)
        self.llm_tuple = None
        if cfg.enable_synthesis_llm:
            try:
                from phase_0_rag_baseline.llm import load_llm

                self.llm_tuple = load_llm(cfg.llm_model)
            except Exception as exc:
                raise RuntimeError(
                    "Phase 2 final-answer LLM failed to load. "
                    "GraphRAG uses the same phase_0_rag_baseline.llm.load_llm path as Phase 0/1. "
                    "Set GRAPH_RAG_ENABLE_SYNTHESIS_LLM=0 only if you explicitly want raw evidence/debug output."
                ) from exc

    @staticmethod
    def _get_query(state: Dict[str, Any]) -> str:
        query = state.get("original_query") or state.get("query") or state.get("input_query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("GraphRAG state must include a non-empty query in 'original_query', 'query', or 'input_query'.")
        return query

    @staticmethod
    def _forced_mode(state: Dict[str, Any]) -> Optional[str]:
        mode = state.get("retrieval_mode", "auto")
        if mode in (None, "", "auto"):
            return None
        if mode not in {"graph", "vector", "timeline", "verification"}:
            raise ValueError(f"Unsupported retrieval_mode: {mode}")
        return str(mode)

    @staticmethod
    def _expand_entities(entities: List[str]) -> List[str]:
        expanded: List[str] = []
        seen = set()

        def add(value: str) -> None:
            value = str(value or "").strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                expanded.append(value)

        for entity in entities:
            add(entity)
        return expanded

    def route(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._get_query(state)
        entities = infer_entities_from_query(query)
        forced_mode = self._forced_mode(state)

        if forced_mode:
            return {
                "intent": forced_mode,
                "intent_rationale": f"Forced retrieval mode: {forced_mode}",
                "entities": [{"name": e} for e in entities],
                "use_vector": forced_mode in {"vector", "verification"},
                "use_graph": forced_mode in {"graph", "verification"},
                "use_timeline": forced_mode in {"timeline", "verification"},
                "use_verification": forced_mode == "verification",
                "step_count": state.get("step_count", 0) + 1,
            }

        decision = self.router.classify(query)

        return {
            "intent": decision.intent,
            "intent_rationale": decision.rationale,
            "entities": [{"name": e} for e in entities],
            "use_vector": decision.use_vector,
            "use_graph": decision.use_graph,
            "use_timeline": decision.use_timeline,
            "use_verification": decision.use_verification,
            "step_count": state.get("step_count", 0) + 1,
        }

    def _build_cypher(self, entities: List[str], relations: List[str], year: Optional[int], limit: int) -> str:
        """Build a Cypher query for graph retrieval.

        IMPORTANT: WHERE must come before OPTIONAL MATCH.  In Neo4j, placing a
        WHERE clause *after* OPTIONAL MATCH makes it filter the optional-match
        results only — the mandatory MATCH returns all edges unfiltered.  Moving
        WHERE before the OPTIONAL MATCHes (via an intermediate WITH) ensures the
        entity/relation filters are applied first, then the optional year lookup
        runs only on the filtered rows.
        """
        entity = entities[0].lower() if entities else ""
        entity_clause = (
            f"AND (toLower(n.name) CONTAINS '{entity}' OR toLower(m.name) CONTAINS '{entity}')"
            if entity else ""
        )
        year_clause = """
            $year IS NULL
            OR ny.value = $year OR my.value = $year
            OR (r.year IS NOT NULL AND r.year = $year)
            OR (n.year IS NOT NULL AND n.year = $year)
            OR (m.year IS NOT NULL AND m.year = $year)
        """.strip()

        cypher = f"""
        MATCH (n)-[r]->(m)
        WHERE type(r) IN $relations
          {entity_clause}
        WITH n, r, m
        OPTIONAL MATCH (n)-[:`{RELATION_TYPES['occurred_in']}`|`{RELATION_TYPES['happened_in']}`]->(ny:Year)
        WITH n, r, m, ny
        OPTIONAL MATCH (m)-[:`{RELATION_TYPES['occurred_in']}`|`{RELATION_TYPES['happened_in']}`]->(my:Year)
        WITH n, r, m, ny, my
        WHERE {year_clause}
        RETURN n.name          AS source_name,
               labels(n)       AS source_labels,
               properties(n)   AS source_properties,
               type(r)         AS relation,
               m.name          AS target_name,
               labels(m)       AS target_labels,
               properties(m)   AS target_properties,
               properties(r)   AS properties,
               coalesce(ny.value, my.value,
                        CASE WHEN r.year IS NOT NULL THEN r.year END,
                        CASE WHEN n.year IS NOT NULL THEN n.year END,
                        CASE WHEN m.year IS NOT NULL THEN m.year END) AS year
        ORDER BY year DESC NULLS LAST, source_name ASC, relation ASC, target_name ASC
        LIMIT $limit
        """
        return cypher.strip()

    def graph_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._get_query(state)
        raw_entities = [e["name"] for e in state.get("entities", [])]
        entities = self._expand_entities(raw_entities)
        intent = state.get("intent", "graph")

        if intent == "timeline":
            return {"graph_results": {"rows": []}}

        # Base relations for historical queries
        base_relations = [
            "FOUGHT_AGAINST", "FOUGHT", "ALLIED_WITH", "SUCCEEDED_BY",
            "RULED", "WROTE_ABOUT", "OCCURRED_IN", "CONTEMPORARY_OF",
            "ASSOCIATED_WITH", "MEMBER_OF", "FOUNDED", "LED"
        ]
        # Modern relations for 19th/20th century
        modern_relations = [
            "SIGNED", "PROTESTED", "IMPRISONED", "ASSASSINATED", "DEMANDED",
            "ESTABLISHED", "INFLUENCED", "RELATED_TO", "JOINED", "PRESIDED_OVER",
            "NEGOTIATED", "CONQUERED", "CAPTURED", "ANNEXED", "INVADED", "BESIEGED",
            "RAIDED", "EXPANDED_INTO", "CONTROLLED", "GOVERNED", "ADMINISTERED",
            "PATRONIZED", "BUILT", "COMMISSIONED", "CONVERTED", "REBELLED_AGAINST",
            "TRIBUTARY_TO", "VASSAL_OF", "MARRIED_TO", "OPPOSED", "VANQUISHED",
            "DEFEATED_BY"
        ]

        relations = base_relations + modern_relations

        import re
        year = None
        match = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", query)
        if match:
            year = int(match.group(1))

        seed_entities = entities or raw_entities
        subgraph = self.graph_store.subgraph(seed_entities, hops=1, limit=self.cfg.graph_top_k * 5)

        rows: List[Dict[str, Any]] = []
        for edge in subgraph.get("edges", []):
            relation = edge.get("relation")
            if relation not in relations:
                continue
            props = edge.get("properties") or {}
            if year is not None:
                edge_year = props.get("year") or props.get("page_year")
                if edge_year is not None and str(edge_year) != str(year):
                    continue
            rows.append(
                {
                    "source_name": edge.get("source_name"),
                    "source_labels": [edge.get("source_label")] if edge.get("source_label") else [],
                    "source_properties": {},
                    "relation": relation,
                    "target_name": edge.get("target_name"),
                    "target_labels": [edge.get("target_label")] if edge.get("target_label") else [],
                    "target_properties": {},
                    "properties": props,
                    "year": props.get("year") or props.get("page_year"),
                }
            )
            if len(rows) >= self.cfg.graph_top_k:
                break

        cypher = (
            "// entity-centered subgraph retrieval; no direct Cypher traversal used\n"
            f"// entities={seed_entities!r}, relations={relations!r}, year={year!r}, limit={self.cfg.graph_top_k}"
        )
        print("\n=== GRAPH RETRIEVAL ===")
        print(cypher)
        print("Params:", {"entities": seed_entities, "relations": relations, "year": year, "limit": self.cfg.graph_top_k})
        return {"cypher_query": cypher, "graph_results": {"rows": rows, "mode": intent}}

    def timeline_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import re

        query = self._get_query(state)
        years = [int(y) for y in re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", query)]
        entities = [e["name"] for e in state.get("entities", [])]

        if years:
            start = min(years)
            end = max(years)
            df = self.timeline_store.search_year_range(start, end, limit=self.cfg.timeline_top_k)
        else:
            keyword = entities[0] if entities else (query.split()[0] if query.split() else query)
            df = self.timeline_store.search_keyword(keyword, limit=self.cfg.timeline_top_k)

        return {
            "sql_query": "SELECT * FROM timeline_facts ...",
            "timeline_results": {"rows": df.to_dict(orient="records")},
        }

    def vector_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("rewritten_query") or self._get_query(state)
        result = self.vector_tool.search(query, k=self.cfg.vector_top_k)
        return {"vector_query": query, "vector_results": result}

    def verify(self, state: Dict[str, Any]) -> Dict[str, Any]:
        graph_rows = state.get("graph_results", {}).get("rows", [])
        vector_rows = state.get("vector_results", {}).get("results", [])
        timeline_rows = state.get("timeline_results", {}).get("rows", [])
        verdict = compare_sources(graph_rows, vector_rows, timeline_rows)
        return {"verification_results": verdict}

    def synthesize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        answer = synthesize_answer(
            state,
            llm_tuple=self.llm_tuple,
            answer_style=state.get("answer_style", "concise"),
            max_words=state.get("max_words", 300),
        )
        return {"final_answer": answer}

    def _flush_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        extracted_records = extract_facts_from_records(batch)

        all_nodes: List[Dict[str, Any]] = []
        all_edges: List[Dict[str, Any]] = []
        for item in extracted_records:
            if not isinstance(item, dict):
                continue
            all_nodes.extend(item.get("nodes", []))
            all_edges.extend(item.get("edges", []))

        node_map: Dict[tuple, Dict[str, Any]] = {}
        for n in all_nodes:
            key = (n["label"], n["name"])
            if key not in node_map:
                node_map[key] = dict(n)
            else:
                node_map[key]["properties"].update(n.get("properties") or {})
        unique_nodes = list(node_map.values())

        self.graph_store.ingest_batch(unique_nodes, all_edges)
        self.timeline_store.ingest_rows(build_fact_table(batch, extracted_records))
        return {"records": len(batch), "nodes": len(unique_nodes), "edges": len(all_edges)}

    def ingest_documents(self, records: List[Dict[str, Any]], reset: bool = False) -> Dict[str, Any]:
        if reset:
            self.graph_store.clear()
            self.timeline_store.reset()
        stats = self._flush_batch(records)
        stats["records"] = len(records)
        return stats

    def ingest_documents_stream(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        reset: bool = False,
        batch_size: int = 50,
        progress=None,
        total_records: Optional[int] = None,
    ) -> Dict[str, Any]:
        if reset:
            self.graph_store.clear()
            self.timeline_store.reset()

        batch: List[Dict[str, Any]] = []
        totals = {"records": 0, "nodes": 0, "edges": 0}

        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                stats = self._flush_batch(batch)
                for k in totals:
                    totals[k] += stats[k]
                if progress is not None:
                    progress.update(len(batch))
                    progress.set_postfix(records=totals["records"], nodes=totals["nodes"], edges=totals["edges"])
                batch = []

        if batch:
            stats = self._flush_batch(batch)
            for k in totals:
                totals[k] += stats[k]
            if progress is not None:
                progress.update(len(batch))
                progress.set_postfix(records=totals["records"], nodes=totals["nodes"], edges=totals["edges"])

        if progress is not None:
            progress.close()
        return totals

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(state)
        out.update(self.route(out))

        intent = out.get("intent")
        if intent == "graph":
            out.update(self.graph_search(out))
            if not self._forced_mode(out) and not out.get("graph_results", {}).get("rows"):
                out.update(self.vector_search(out))
        elif intent == "timeline":
            out.update(self.timeline_search(out))
            if not self._forced_mode(out) and not out.get("timeline_results", {}).get("rows"):
                out.update(self.vector_search(out))
        elif intent == "verification":
            out.update(self.graph_search(out))
            out.update(self.vector_search(out))
            out.update(self.timeline_search(out))
            out.update(self.verify(out))
        else:
            out.update(self.vector_search(out))

        out.update(self.synthesize(out))
        return out


class _FallbackApp:
    def __init__(self, tools: GraphRAGTools):
        self.tools = tools

    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self.tools.execute(state)


def build_graph_rag_app(cfg: GraphRAGConfig, graph_store: GraphStore):
    tools = GraphRAGTools(cfg, graph_store)

    if not _HAS_LANGGRAPH:
        return _FallbackApp(tools), tools

    workflow = StateGraph(GraphRAGStateDict)
    workflow.add_node("router", tools.route)
    workflow.add_node("vector_search", tools.vector_search)
    workflow.add_node("graph_search", tools.graph_search)
    workflow.add_node("timeline_search", tools.timeline_search)
    workflow.add_node("verify", tools.verify)
    workflow.add_node("synthesize", tools.synthesize)

    def route_after_router(state):
        intent = state.get("intent", "vector")
        if intent == "graph":
            return "graph_search"
        if intent == "timeline":
            return "timeline_search"
        if intent == "verification":
            return "graph_search"
        return "vector_search"

    def route_after_graph(state):
        if state.get("intent") == "verification":
            return "vector_search"
        if tools._forced_mode(state) == "graph":
            return "synthesize"
        if not state.get("graph_results", {}).get("rows"):
            return "vector_search"
        return "synthesize"

    def route_after_vector(state):
        if state.get("intent") == "verification":
            return "timeline_search"
        return "synthesize"

    def route_after_timeline(state):
        if state.get("intent") == "verification":
            return "verify"
        if tools._forced_mode(state) == "timeline":
            return "synthesize"
        return "synthesize"

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_after_router)
    workflow.add_conditional_edges("graph_search", route_after_graph)
    workflow.add_conditional_edges("vector_search", route_after_vector)
    workflow.add_conditional_edges("timeline_search", route_after_timeline)
    workflow.add_edge("verify", "synthesize")
    workflow.add_edge("synthesize", END)
    app = workflow.compile()
    return app, tools