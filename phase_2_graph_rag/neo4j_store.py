"""Neo4j graph store with an in-memory fallback.

The default path now performs real Neo4j writes in batches using UNWIND.
A memory backend is still kept for local smoke tests and environments where
Neo4j is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import NODE_LABELS, RELATION_TYPES
from .utils import normalize_name

try:
    from neo4j import GraphDatabase  # type: ignore
    _HAS_NEO4J = True
except Exception:
    GraphDatabase = None
    _HAS_NEO4J = False


@dataclass
class QuerySpec:
    mode: str
    cypher: str
    params: Dict[str, Any]
    entities: List[str]
    relations: List[str]
    year: Optional[int] = None
    limit: int = 20


class MemoryGraphBackend:
    def __init__(self):
        self.nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()

    def upsert_node(self, label: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        key = (label, name)
        node = self.nodes.get(key)
        if node is None:
            node = {"label": label, "name": name, "properties": {}}
            self.nodes[key] = node
        if properties:
            node["properties"].update(properties)

    def upsert_edge(
        self,
        src_label: str,
        src_name: str,
        relation: str,
        tgt_label: str,
        tgt_name: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.upsert_node(src_label, src_name)
        self.upsert_node(tgt_label, tgt_name)
        edge = {
            "source_label": src_label,
            "source_name": src_name,
            "relation": relation,
            "target_label": tgt_label,
            "target_name": tgt_name,
            "properties": properties or {},
        }
        for existing in self.edges:
            if (
                existing["source_label"] == edge["source_label"]
                and existing["source_name"] == edge["source_name"]
                and existing["relation"] == edge["relation"]
                and existing["target_label"] == edge["target_label"]
                and existing["target_name"] == edge["target_name"]
            ):
                existing["properties"].update(edge["properties"])
                return
        self.edges.append(edge)

    def ingest(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        for n in nodes:
            self.upsert_node(n["label"], n["name"], n.get("properties"))
        for e in edges:
            self.upsert_edge(
                e["source_label"], e["source_name"], e["relation"],
                e["target_label"], e["target_name"], e.get("properties"),
            )

    def ingest_batch(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.ingest(nodes, edges)

    def schema_overview(self) -> Dict[str, Any]:
        labels = sorted({label for label, _ in self.nodes.keys()})
        rels = sorted({e["relation"] for e in self.edges})
        return {
            "labels": labels,
            "relations": rels,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }

    @staticmethod
    def _entity_matches(name: str, entities_n: List[str]) -> bool:
        name_n = normalize_name(name)
        if not name_n or not entities_n:
            return False
        return any(entity == name_n or entity in name_n or name_n in entity for entity in entities_n)

    def query_spec(self, spec: QuerySpec) -> List[Dict[str, Any]]:
        entities_n = [normalize_name(e) for e in spec.entities if e]
        results: List[Dict[str, Any]] = []

        if spec.mode == "timeline":
            for e in self.edges:
                if e["relation"] not in {RELATION_TYPES["occurred_in"], RELATION_TYPES["happened_in"]}:
                    continue
                if spec.year is not None and normalize_name(e["target_name"]) != normalize_name(str(spec.year)):
                    continue
                if entities_n and not (
                    self._entity_matches(e["source_name"], entities_n)
                    or self._entity_matches(e["target_name"], entities_n)
                ):
                    continue
                results.append(e)
            return results[: spec.limit]

        if spec.mode == "verification":
            for e in self.edges:
                if entities_n and not (
                    self._entity_matches(e["source_name"], entities_n)
                    or self._entity_matches(e["target_name"], entities_n)
                ):
                    continue
                if spec.relations and e["relation"] not in spec.relations:
                    continue
                results.append(e)
            return results[: spec.limit]

        for e in self.edges:
            if spec.relations and e["relation"] not in spec.relations:
                continue
            if entities_n and not (
                self._entity_matches(e["source_name"], entities_n)
                or self._entity_matches(e["target_name"], entities_n)
            ):
                continue
            if spec.year is not None:
                props = e.get("properties") or {}
                prop_year = props.get("year") or props.get("page_year")
                if prop_year is not None and str(prop_year) != str(spec.year):
                    continue
            results.append(e)
        return results[: spec.limit]

    def subgraph(self, center_entities: List[str], hops: int = 1, limit: int = 50) -> Dict[str, Any]:
        entities_n = {normalize_name(e) for e in center_entities if e}
        nodes = []
        edges = []

        def matches(name: str) -> bool:
            name_n = normalize_name(name)
            return bool(name_n and any(entity == name_n or entity in name_n or name_n in entity for entity in entities_n))

        for (label, name), node in self.nodes.items():
            if matches(name):
                nodes.append(node)
        frontier = set(entities_n)
        for _ in range(hops):
            new_frontier = set()
            for e in self.edges:
                s = normalize_name(e["source_name"])
                t = normalize_name(e["target_name"])
                if s in frontier or t in frontier:
                    edges.append(e)
                    for key in [(e["source_label"], e["source_name"]), (e["target_label"], e["target_name"] )]:
                        node = self.nodes.get(key)
                        if node and node not in nodes:
                            nodes.append(node)
                        new_frontier.add(normalize_name(key[1]))
            frontier = new_frontier
        return {"nodes": nodes[:limit], "edges": edges[:limit]}


class Neo4jBackend:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if not _HAS_NEO4J:
            raise ImportError("neo4j package is not installed")
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self._test_connection()
        self.ensure_constraints()

    def _test_connection(self):
        with self._session() as session:
            result = session.run("RETURN 1 AS test")
            print("Neo4j connection OK, database:", self.database)

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    @staticmethod
    def _write(session, func, *args, **kwargs):
        if hasattr(session, "execute_write"):
            return session.execute_write(func, *args, **kwargs)
        return session.write_transaction(func, *args, **kwargs)

    def ensure_constraints(self) -> None:
        constraints = []
        for label in dict.fromkeys(NODE_LABELS.values()):
            constraints.append(
                f"CREATE CONSTRAINT {label.lower()}_name_unique IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.name IS UNIQUE"
            )
        with self._session() as session:
            for cypher in constraints:
                session.run(cypher).consume()

    def clear(self) -> None:
        cypher = "MATCH (n) DETACH DELETE n"
        with self._session() as session:
            session.run(cypher).consume()

    @staticmethod
    def _clean_props(properties: Optional[Dict[str, Any]], *, name: str) -> Dict[str, Any]:
        props = dict(properties or {})
        props.pop("name", None)
        props.pop("label", None)
        props["name"] = name
        return props

    def _upsert_nodes_tx(self, tx, label: str, rows: List[Dict[str, Any]]):
        cypher = f"""
        UNWIND $rows AS row
        MERGE (n:`{label}` {{name: row.name}})
        SET n += row.properties
        RETURN count(n) AS merged
        """
        tx.run(cypher, rows=rows).consume()

    def _upsert_edges_tx(self, tx, src_label: str, rel: str, tgt_label: str, rows: List[Dict[str, Any]]):
        cypher = f"""
        UNWIND $rows AS row
        MERGE (a:`{src_label}` {{name: row.src_name}})
        MERGE (b:`{tgt_label}` {{name: row.tgt_name}})
        MERGE (a)-[r:`{rel}`]->(b)
        SET r += row.properties
        RETURN count(r) AS merged
        """
        tx.run(cypher, rows=rows).consume()

    def ingest_batch(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        node_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for n in nodes:
            node_groups[n["label"]].append({
                "name": n["name"],
                "properties": self._clean_props(n.get("properties"), name=n["name"]),
            })

        edge_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            edge_groups[(e["source_label"], e["relation"], e["target_label"])].append({
                "src_name": e["source_name"],
                "tgt_name": e["target_name"],
                "properties": e.get("properties") or {},
            })

        with self._session() as session:
            for label, rows in node_groups.items():
                if not rows:
                    continue
                self._write(session, self._upsert_nodes_tx, label, rows)
            for (src_label, rel, tgt_label), rows in edge_groups.items():
                if not rows:
                    continue
                self._write(session, self._upsert_edges_tx, src_label, rel, tgt_label, rows)

    def ingest(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.ingest_batch(nodes, edges)

    def schema_overview(self) -> Dict[str, Any]:
        cypher = """
        CALL db.schema.visualization()
        YIELD nodes, relationships
        RETURN nodes, relationships
        """
        with self._session() as session:
            records = session.run(cypher)
            return {"records": [r.data() for r in records]}

    def query_spec(self, spec: QuerySpec) -> List[Dict[str, Any]]:
        with self._session() as session:
            records = session.run(spec.cypher, **spec.params)
            return [r.data() for r in records]

    def subgraph(self, center_entities: List[str], hops: int = 2, limit: int = 150) -> Dict[str, Any]:
        hops = max(1, min(int(hops), 3))

        cypher = f"""
        MATCH p=(n)-[*1..{hops}]-(m)

        WHERE any(
            entity IN $entities
            WHERE toLower(n.name) CONTAINS toLower(entity)
            OR toLower(m.name) CONTAINS toLower(entity)
        )

        UNWIND relationships(p) AS r

        RETURN DISTINCT
            startNode(r).name AS source_name,
            head(labels(startNode(r))) AS source_label,

            type(r) AS relation,

            endNode(r).name AS target_name,
            head(labels(endNode(r))) AS target_label

        LIMIT $limit
        """
        with self._session() as session:
            records = session.run(cypher, entities=center_entities, limit=limit)
            edges = [r.data() for r in records]

            nodes_seen = set()
            nodes = []

            for e in edges:

                src_key = (e["source_label"], e["source_name"])
                tgt_key = (e["target_label"], e["target_name"])

                if src_key not in nodes_seen:
                    nodes_seen.add(src_key)
                    nodes.append(
                        {
                            "label": e["source_label"],
                            "name": e["source_name"],
                        }
                    )

                if tgt_key not in nodes_seen:
                    nodes_seen.add(tgt_key)
                    nodes.append(
                        {
                            "label": e["target_label"],
                            "name": e["target_name"],
                        }
                    )

            return {
                "nodes": nodes,
                "edges": edges,
            }


class GraphStore:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        use_memory_fallback: bool = True,
    ):
        self.database = database
        self.use_memory_fallback = use_memory_fallback
        self.backend = None
        self.memory = MemoryGraphBackend()

        if _HAS_NEO4J:
            try:
                self.backend = Neo4jBackend(uri, user, password, database=database)
            except Exception:
                if not use_memory_fallback:
                    raise
                self.backend = None
        elif not use_memory_fallback:
            raise ImportError("neo4j package is unavailable and memory fallback disabled")

    @property
    def is_neo4j(self) -> bool:
        return self.backend is not None

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()

    def clear(self) -> None:
        if self.backend is not None:
            self.backend.clear()
        self.memory.clear()

    def ingest_batch(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.memory.ingest_batch(nodes, edges)
        if self.backend is not None:
            self.backend.ingest_batch(nodes, edges)

    def ingest(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self.ingest_batch(nodes, edges)

    def query_spec(self, spec: QuerySpec) -> List[Dict[str, Any]]:
        if self.backend is not None:
            try:
                return self.backend.query_spec(spec)
            except Exception:
                if not self.use_memory_fallback:
                    raise
        return self.memory.query_spec(spec)

    def run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Compatibility helper used by legacy reasoning code."""
        params = params or {}
        spec = QuerySpec(
            mode="custom",
            cypher=cypher,
            params=params,
            entities=list(params.get("entities") or []),
            relations=list(params.get("relations") or []),
            year=params.get("year"),
            limit=int(params.get("limit", 50)),
        )
        return self.query_spec(spec)

    def subgraph(self, center_entities: List[str], hops: int = 1, limit: int = 50) -> Dict[str, Any]:
        if self.backend is not None:
            try:
                return self.backend.subgraph(center_entities, hops=hops, limit=limit)
            except Exception:
                if not self.use_memory_fallback:
                    raise
        return self.memory.subgraph(center_entities, hops=hops, limit=limit)

    def schema_overview(self) -> Dict[str, Any]:
        if self.backend is not None:
            try:
                return self.backend.schema_overview()
            except Exception:
                if not self.use_memory_fallback:
                    raise
        return self.memory.schema_overview()
