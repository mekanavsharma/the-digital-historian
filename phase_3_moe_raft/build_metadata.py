# phase_3_moe_raft/build_metadata.py

"""
Aggregate all raw chunk JSON files (e.g., 112 files) into a single
chunks_meta.json list.  The order of chunks must exactly match the
order used to build the BM25 and FAISS indexes.
"""
import json
from pathlib import Path
from phase_3_moe_raft.config import RAW_JSON_DIR, PHASE0_CHUNKS_META


def load_json_file(path: Path):
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        chunks = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {lineno} in {path}: {exc.msg}"
                    ) from exc
        return chunks

    raise ValueError(f"Unsupported file extension: {path.suffix}")


def gather_chunks():
    all_chunks = []
    raw_dir = Path(RAW_JSON_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW_JSON_DIR does not exist: {raw_dir}")

    json_files = sorted(raw_dir.rglob("*.json*"))
    print(f"Found {len(json_files)} JSON/JSONL files.")
    for fp in json_files:
        all_chunks.extend(load_json_file(fp))

    for ch in all_chunks:
        ch.setdefault("expert_domain", "Modern")
        ch.setdefault("historian_perspective", "Nationalist")

    output_path = Path(PHASE0_CHUNKS_META)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_chunks)} chunks to {output_path}")

if __name__ == "__main__":
    gather_chunks()