"""Debug script to see what's actually in your source text and what gets extracted."""

import json
import re
import sys
from pathlib import Path

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except:
    nlp = None
    print("WARNING: spaCy not loaded")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phase_2_graph_rag.extractor import extract_facts_from_record


def find_source_files(documents_path: str):
    """Find all JSONL files."""
    path = Path(documents_path)
    if path.is_file():
        return [path]
    return sorted(path.glob("*.jsonl"))


def debug_documents(documents_path: str, keywords: list, max_chunks: int = 50):
    """Search for chunks containing keywords and show extraction results."""

    files = find_source_files(documents_path)
    if not files:
        print(f"No JSONL files found at {documents_path}")
        return

    found_chunks = []

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except:
                    continue

                content = record.get("content", "")
                content_lower = content.lower()

                # Check if any keyword is in this chunk
                for kw in keywords:
                    if kw.lower() in content_lower:
                        found_chunks.append({
                            "file": fpath.name,
                            "line": line_num,
                            "content": content,
                            "record": record,
                            "matched_keyword": kw
                        })
                        break

                if len(found_chunks) >= max_chunks:
                    break
        if len(found_chunks) >= max_chunks:
            break

    print(f"\n{'='*80}")
    print(f"Found {len(found_chunks)} chunks containing: {keywords}")
    print(f"{'='*80}\n")

    for i, chunk in enumerate(found_chunks):
        print(f"\n{'─'*80}")
        print(f"CHUNK {i+1} | File: {chunk['file']} | Line: {chunk['line']} | Keyword: {chunk['matched_keyword']}")
        print(f"{'─'*80}")

        content = chunk["content"]

        # Show first 800 chars of content
        print(f"\n📖 CONTENT (first 800 chars):\n{content[:800]}")
        if len(content) > 800:
            print(f"... [{len(content) - 800} more chars]")

        # Show spaCy NER results
        if nlp:
            print(f"\n🔍 spaCy NER entities:")
            doc = nlp(content)
            for ent in doc.ents:
                print(f"   • {ent.text:40s} → {ent.label_}")

        # Show capitalized phrases
        print(f"\n📝 Capitalized multi-word phrases:")
        phrases = set(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", content))
        for p in sorted(phrases):
            print(f"   • {p}")

        # Show what extractor produces
        print(f"\n⚙️ EXTRACTOR OUTPUT:")
        nodes, edges = extract_facts_from_record(chunk["record"])

        print(f"   Nodes ({len(nodes)}):")
        person_nodes = [n for n in nodes if n["label"] == "Person"]
        org_nodes = [n for n in nodes if n["label"] == "Organization"]
        other_nodes = [n for n in nodes if n["label"] not in ("Person", "Organization", "Year", "Passage", "Source")]

        if person_nodes:
            print(f"     Persons:")
            for n in person_nodes:
                print(f"       • {n['name']} (aliases: {n['properties'].get('aliases', [])})")
        if org_nodes:
            print(f"     Organizations:")
            for n in org_nodes:
                print(f"       • {n['name']} (aliases: {n['properties'].get('aliases', [])})")
        if other_nodes:
            print(f"     Other:")
            for n in other_nodes[:10]:
                print(f"       • [{n['label']}] {n['name']}")

        print(f"   Edges ({len(edges)}):")
        # Filter to interesting relations
        interesting = ["FOUNDED", "MEMBER_OF", "LED", "ASSOCIATED_WITH", "JOINED", "ESTABLISHED"]
        interesting_edges = [e for e in edges if e["relation"] in interesting]
        other_edges = [e for e in edges if e["relation"] not in interesting]

        if interesting_edges:
            print(f"     KEY RELATIONSHIPS:")
            for e in interesting_edges:
                print(f"       • {e['source_name']} --{e['relation']}--> {e['target_name']}")
        if other_edges:
            print(f"     Other ({len(other_edges)}):")
            for e in other_edges[:5]:
                print(f"       • {e['source_name']} --{e['relation']}--> {e['target_name']}")
            if len(other_edges) > 5:
                print(f"       ... and {len(other_edges) - 5} more")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents-path", type=str, required=True)
    parser.add_argument("--keywords", type=str, nargs="+", default=["Savarkar", "Satis Chandra Mukherjee", "Abhinav"])
    args = parser.parse_args()

    debug_documents(args.documents_path, args.keywords)