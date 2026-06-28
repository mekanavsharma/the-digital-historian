"""CLI for phase 2 GraphRAG – rule‑based extraction only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterator

from .config import GraphRAGConfig
from .graph import build_graph_rag_app
from .neo4j_store import GraphStore
from .utils import normalize_text

try:
    from tqdm.auto import tqdm
    _HAS_TQDM = True
except Exception:
    tqdm = None
    _HAS_TQDM = False


def count_records(documents_path: str) -> int:
    path = Path(documents_path)
    if not path.exists():
        return 0
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    total = 0
    for fn in sorted(path.iterdir()):
        if fn.suffix.lower() != ".jsonl":
            continue
        with fn.open("r", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def iter_records(documents_path: str, limit: int | None = None) -> Iterator[Dict[str, Any]]:
    import json as _json
    path = Path(documents_path)
    yielded = 0
    if path.is_file():
        files = [path]
    else:
        files = [fn for fn in sorted(path.iterdir()) if fn.suffix.lower() == ".jsonl"] if path.exists() else []
    for fn in files:
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = _json.loads(line)
                except Exception:
                    continue
                yield {
                    "content": row.get("content", ""),
                    "metadata": row,
                }
                yielded += 1
                if limit is not None and yielded >= limit:
                    return


def main():
    parser = argparse.ArgumentParser(description="Phase 2 GraphRAG – rule‑based extraction")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--show-schema", action="store_true")
    parser.add_argument("--visualize", type=str, default=None)
    parser.add_argument("--visualize-query", action="store_true")
    parser.add_argument("--retrieval-mode", choices=["auto", "graph", "vector", "timeline", "verification"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--documents-path", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ingest-batch-size", type=int, default=200)
    parser.add_argument("--answer-style", type=str, default="concise")
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--neo4j-uri", type=str, default=None)
    parser.add_argument("--neo4j-user", type=str, default=None)
    parser.add_argument("--neo4j-password", type=str, default=None)
    parser.add_argument("--neo4j-database", type=str, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--page-mode", action="store_true", help="Aggregate chunks by page before extraction (rule‑based)")

    args = parser.parse_args()

    cfg = GraphRAGConfig()
    if args.documents_path:
        cfg.documents_path = args.documents_path
    if args.neo4j_uri:
        cfg.neo4j_uri = args.neo4j_uri
    if args.neo4j_user:
        cfg.neo4j_user = args.neo4j_user
    if args.neo4j_password:
        cfg.neo4j_password = args.neo4j_password
    if args.neo4j_database:
        cfg.neo4j_database = args.neo4j_database
    if args.limit is not None:
        cfg.ingest_limit = args.limit

    cfg.reset_graph_on_ingest = bool(args.reset)
    cfg.ensure_dirs()

    store = GraphStore(
        cfg.neo4j_uri,
        cfg.neo4j_user,
        cfg.neo4j_password,
        database=cfg.neo4j_database,
        use_memory_fallback=False,
    )

    app, tools = build_graph_rag_app(cfg, store)

    if args.show_schema:
        print(json.dumps(store.schema_overview(), indent=2, ensure_ascii=False))

    if args.ingest:
        total = count_records(cfg.documents_path)
        if total == 0:
            raise SystemExit(f"No records found at {cfg.documents_path}")
        if args.limit is not None:
            total = min(total, args.limit)

        print(f"Preparing to ingest {total} records from {cfg.documents_path}")
        print(f"Page aggregation: {args.page_mode}")
        print(f"Neo4j: {cfg.neo4j_uri}" if store.is_neo4j else "Neo4j fallback mode (memory)")

        progress = None
        if _HAS_TQDM and not args.no_progress:
            progress = tqdm(total=total, desc="Ingesting", unit="docs")

        records_iter = iter_records(cfg.documents_path, limit=cfg.ingest_limit)

        if args.page_mode:
            from .page_aggregator import ingest_pages_rule_based
            stats = ingest_pages_rule_based(
                records_iter,
                graph_store=store,
                timeline_store=tools.timeline_store,
                batch_size=max(1, args.ingest_batch_size),
                progress=progress,
            )
        else:
            # Original per‑chunk ingestion
            stats = tools.ingest_documents_stream(
                records_iter,
                reset=cfg.reset_graph_on_ingest,
                batch_size=max(1, args.ingest_batch_size),
                progress=progress,
                total_records=total,
            )
        print(json.dumps(stats, indent=2))

    if args.visualize:
        from .visualize import export_subgraph_html
        out_path = export_subgraph_html(
            store,
            center_entities=[args.visualize],
            out_dir=cfg.graph_export_dir,
            file_name=f"{normalize_text(args.visualize).replace(' ', '_')}.html",
        )
        print(f"\nGraph saved to:\n{out_path}")

    if args.query:
        print("\n[GraphRAG] Processing query...\n")
        result = app.invoke({
            "original_query": args.query,
            "retrieval_mode": args.retrieval_mode,
            "answer_style": args.answer_style,
            "max_words": args.max_words,
            "chat_history": [],
            "session_id": "cli",
        })
        print("Result keys:", result.keys())

        print("\n=== FINAL ANSWER ===\n")
        print(result.get("final_answer", ""))
        if args.debug:
            print("\n=== ROUTING INFO ===")
            print("intent:", result.get("intent"))
            print("rationale:", result.get("intent_rationale"))
            if result.get("cypher_query"):
                print("\n=== CYPHER ===\n")
                print(result.get("cypher_query"))
            if result.get("sql_query"):
                print("\n=== SQL ===\n")
                print(result.get("sql_query"))

    store.close()


if __name__ == "__main__":
    main()