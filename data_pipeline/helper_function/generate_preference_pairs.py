import json
import os
import time
import hashlib
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------------------------------------------------------------ #
# Kaggle auth + upload helper
# ------------------------------------------------------------------ #
from kaggle_secrets import UserSecretsClient
from kaggle.api.kaggle_api_extended import KaggleApi

user_secrets = UserSecretsClient()
os.environ['KAGGLE_USERNAME'] = 'kanav608'
os.environ['KAGGLE_KEY'] = 'KGAT_c97f2d5fb9093988f33dadee50e869bb'  # set this under Notebook -> Add-ons -> Secrets

api = KaggleApi()
api.authenticate()

KAGGLE_DATASET = "kanav608/raft-intermediate"
DPO_DATASET_DIR = "/kaggle/working/dpo_outputs"
os.makedirs(DPO_DATASET_DIR, exist_ok=True)


def upload_to_dataset(dir_path, message="Update DPO files"):
    """Commit the given directory as a new version of a Kaggle Dataset.
    Always rewrites dataset-metadata.json so a stale/corrupt leftover can't block future uploads."""
    metadata_path = os.path.join(dir_path, "dataset-metadata.json")
    metadata = {
        "id": KAGGLE_DATASET,
        "title": "RAFT training examples",
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    try:
        api.dataset_create_version(
            folder=dir_path,
            version_notes=message,
            quiet=True,   # quiet=True here — this fires often, no need for verbose output every chunk
        )
        print(f"   📦 Uploaded to Kaggle dataset: {message}")
    except Exception as e:
        # Never let an upload failure kill the generation run — just log and keep going
        print(f"   ⚠️ Upload failed (will retry after next chunk): {e}")


def push_current_output():
    """Copy the growing OUTPUT_FILE into the upload staging dir and push a new dataset version."""
    if not (os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0):
        print("   ⚠️ Nothing to upload yet.")
        return
    dest = os.path.join(DPO_DATASET_DIR, os.path.basename(OUTPUT_FILE))
    shutil.copy(OUTPUT_FILE, dest)
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        total_pairs = sum(1 for _ in f)
    upload_to_dataset(DPO_DATASET_DIR, f"DPO preference pairs — {total_pairs} pairs so far")


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
RAFT_FILE = "/kaggle/input/datasets/kanav608/raft-data/raft_data.jsonl"
OUTPUT_FILE = "/kaggle/working/preference_pairs.jsonl"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 120
MAX_MODEL_LEN = 4096
CHUNK_SIZE = 2000
SAFETY_MARGIN = MAX_NEW_TOKENS + 50
TENSOR_PARALLEL_SIZE = 2

# ------------------------------------------------------------------ #
# Load tokenizer + vLLM engine (multi-GPU)
# ------------------------------------------------------------------ #
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading vLLM with tensor_parallel_size={TENSOR_PARALLEL_SIZE} ...")
llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.85,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
)
print("✅ vLLM engine loaded on both GPUs.")

# ------------------------------------------------------------------ #
# Helpers (unchanged logic)
# ------------------------------------------------------------------ #
def build_full_prompt(rec):
    docs_block = "\n\n".join(rec["documents"])
    return f"""{rec['instruction']}
{docs_block}
Question: {rec['question']}"""

def build_rejected_gen_prompt(rec):
    docs_block = "\n\n".join(rec["documents"])
    perspective = rec.get("perspective", "")
    historian = rec.get("historian", "")
    hint = f"You are a {perspective} historian ({historian}). " if perspective and historian else ""
    return f"""{hint}Below are some documents and a question about them.
{docs_block}
Question: {rec['question']}
Write a plausible-sounding answer to the question that contains exactly one deliberate factual error (wrong date, wrong person, wrong place, or wrong event), while matching the style and length of a real answer. Output ONLY the answer text — no explanation, no labels, no quotes."""

def fits_context(prompt_text):
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    n = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
    return n <= (MAX_MODEL_LEN - SAFETY_MARGIN)

def prompt_hash(prompt_text):
    return hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:16]

def batch_generate_rejected(prompts, temperature=0.8, max_tokens=MAX_NEW_TOKENS):
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    params = SamplingParams(temperature=temperature, top_p=0.9, max_tokens=max_tokens)
    outputs = llm.generate(formatted, params, use_tqdm=True)
    return [o.outputs[0].text.strip().strip('"') for o in outputs]

def generate_chunk_safe(chunk):
    """Try a chunk; on failure bisect and retry so only the true bad record(s) get dropped."""
    if not chunk:
        return []
    try:
        prompts = [build_rejected_gen_prompt(r) for r in chunk]
        rejected = batch_generate_rejected(prompts)
        return list(zip(chunk, rejected))
    except Exception as e:
        if len(chunk) == 1:
            print(f"   ⚠️ Dropping 1 unrecoverable record (historian={chunk[0].get('historian')}): {e}")
            return []
        mid = len(chunk) // 2
        return generate_chunk_safe(chunk[:mid]) + generate_chunk_safe(chunk[mid:])

# ------------------------------------------------------------------ #
# Main conversion loop
# ------------------------------------------------------------------ #
with open(RAFT_FILE, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]
print(f"Loaded {len(records)} RAFT records.")

already_done = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                already_done.add(json.loads(line).get("_qid"))
print(f"Already have {len(already_done)} pairs from previous run — will skip those.")

remaining = []
dropped_oversized = 0
for r in records:
    fp = build_full_prompt(r)
    qid = prompt_hash(fp)
    if qid in already_done:
        continue
    if not fits_context(build_rejected_gen_prompt(r)):
        dropped_oversized += 1
        continue
    remaining.append((qid, r))

print(f"Dropped {dropped_oversized} oversized records.")
print(f"{len(remaining)} records left to process.\n")

t0 = time.time()
with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk_items = remaining[start : start + CHUNK_SIZE]
        chunk = [r for _, r in chunk_items]
        print(f"--- Chunk {start}-{start + len(chunk)} / {len(remaining)} ---")

        results = generate_chunk_safe(chunk)

        count = 0
        for rec, rejected in results:
            chosen = rec["output"].strip()
            if not rejected or not chosen or rejected.strip().lower() == chosen.lower():
                continue
            fp = build_full_prompt(rec)
            pair = {
                "prompt": fp,
                "chosen": chosen,
                "rejected": rejected,
                "source": rec.get("historian", "unknown"),
                "_qid": prompt_hash(fp),
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            fout.flush()
            count += 1
        print(f"   → wrote {count} pairs this chunk")

        # Push progress to Kaggle the moment this chunk is done — never lose more than one chunk's work
        fout.flush()
        os.fsync(fout.fileno())
        push_current_output()

print(f"\n🎉 Done in {time.time() - t0:.1f}s.")