"""
Graph visualization utilities.
"""

from __future__ import annotations

from pathlib import Path
from pyvis.network import Network


def export_rows_html(
    rows,
    out_dir: str,
    file_name: str = "graph.html",
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    out_path = str(Path(out_dir) / file_name)

    net = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
    )

    added_nodes = set()

    for row in rows:

        src = row.get("source_name")
        tgt = row.get("target_name")
        rel = row.get("relation")

        src_label = (
            row.get("source_labels", ["Node"])[0]
            if row.get("source_labels")
            else "Node"
        )

        tgt_label = (
            row.get("target_labels", ["Node"])[0]
            if row.get("target_labels")
            else "Node"
        )

        if src and src not in added_nodes:
            net.add_node(
                src,
                label=src,
                title=src_label,
            )
            added_nodes.add(src)

        if tgt and tgt not in added_nodes:
            net.add_node(
                tgt,
                label=tgt,
                title=tgt_label,
            )
            added_nodes.add(tgt)

        if src and tgt:
            net.add_edge(
                src,
                tgt,
                title=rel,
                label=rel,
            )

    net.show_buttons(filter_=["physics"])

    net.save_graph(out_path)

    return out_path


def export_subgraph_html(
    store,
    center_entities,
    out_dir,
    file_name="subgraph.html",
):

    sub = store.subgraph(
        center_entities,
        hops=2,
        limit=150,
    )

    rows = []

    for edge in sub.get("edges", []):

        rows.append(
            {
                "source_name": edge["source_name"],
                "target_name": edge["target_name"],
                "relation": edge["relation"],
                "source_labels": [edge["source_label"]],
                "target_labels": [edge["target_label"]],
            }
        )

    return export_rows_html(
        rows,
        out_dir,
        file_name,
    )