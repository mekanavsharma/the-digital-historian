# ============================================================
# Preference Pair Cleaner + Fixer  (dual-T4 / tensor parallel)
# ============================================================
# !pip install -q vllm spacy
# !python -m spacy download en_core_web_sm -q

import json
import os
import hashlib
import random
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import spacy

# ------------------------------------------------------------------ #
# Paths & Config
# ------------------------------------------------------------------ #
RAFT_FILE   = "/kaggle/input/datasets/kanav608/raft-data/raft_data.jsonl"
DPO_FILE    = "/kaggle/input/datasets/kanav608/raft-intermediate/preference_pairs.jsonl"

# Write everything under /kaggle/working (input is read-only)
FIXED_FILE  = "/kaggle/working/preference_pairs_fixed.jsonl"
FINAL_FILE  = "/kaggle/working/preference_pairs_final.jsonl"

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 120
MAX_MODEL_LEN  = 4096
TENSOR_PARALLEL_SIZE = 2          # both T4s

# ------------------------------------------------------------------ #
# Load models
# ------------------------------------------------------------------ #
print("Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

print(f"Loading tokenizer + vLLM (tensor_parallel_size={TENSOR_PARALLEL_SIZE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

llm = LLM(
    model=MODEL_ID,
    dtype="float16",
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=0.85,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    # Uncomment if you hit CUDA graph / NCCL issues on Kaggle:
    # enforce_eager=True,
    # disable_custom_all_reduce=True,
)
print("✅ Models ready.\n")

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def prompt_hash(prompt_text: str) -> str:
    return hashlib.md5(prompt_text.encode("utf-8")).hexdigest()[:16]

def build_full_prompt(rec: dict) -> str:
    docs_block = "\n\n".join(rec["documents"])
    return f"""{rec['instruction']}
{docs_block}
Question: {rec['question']}"""

def is_meaningfully_different(chosen: str, rejected: str) -> bool:
    def content_words(text):
        return set(
            w.strip('.,;:()"\'').lower()
            for w in text.split()
            if len(w) > 3
        )
    c_words = content_words(chosen)
    r_words = content_words(rejected)
    if not c_words:
        return True
    overlap = len(c_words & r_words) / len(c_words)
    return overlap < 0.75

# Entity-swap pools (Indian history domain)
DATE_POOL  = ["1757", "1857", "1885", "1905", "1919", "1930", "1942", "1947"]
NAME_POOL  = ["Gandhi", "Nehru", "Tilak", "Gokhale", "Bose", "Motilal Nehru", "Irwin", "Jinnah"]
PLACE_POOL = ["Bengal", "Bombay", "Punjab", "Madras", "Delhi", "Calcutta"]

def corrupt_answer(answer: str):
    doc = nlp(answer)
    ents = [e for e in doc.ents if e.label_ in ("DATE", "PERSON", "GPE", "ORG", "LOC")]
    if not ents:
        return None
    target = random.choice(ents)
    if target.label_ == "DATE":
        pool = DATE_POOL
    elif target.label_ == "PERSON":
        pool = NAME_POOL
    else:
        pool = PLACE_POOL
    candidates = [p for p in pool if p.lower() != target.text.lower()]
    if not candidates:
        return None
    replacement = random.choice(candidates)
    return answer[:target.start_char] + replacement + answer[target.end_char:]

def build_rejected_gen_prompt_v2(rec: dict, chosen: str) -> str:
    docs_block = "\n\n".join(rec["documents"])
    return f"""Below are documents and a question about them.
{docs_block}
Question: {rec['question']}
The correct answer is: "{chosen}"
Write a DIFFERENT answer that CONTRADICTS or gets a key fact wrong compared to the correct answer above — not a rewording of it. It must sound plausible and match the style/length of the correct answer, but change what actually happened (wrong date, wrong person, wrong outcome, etc.).
Output ONLY the corrupted answer text."""

def batch_generate_rejected_v2(prompts, temperature=0.9, max_tokens=MAX_NEW_TOKENS):
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

# ------------------------------------------------------------------ #
# 1. Index original RAFT records
# ------------------------------------------------------------------ #
raft_by_qid = {}
with open(RAFT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = prompt_hash(build_full_prompt(rec))
        raft_by_qid[qid] = rec
print(f"Indexed {len(raft_by_qid)} RAFT records.")

# ------------------------------------------------------------------ #
# 2. Separate good vs bad pairs
# ------------------------------------------------------------------ #
good_pairs, bad_pairs = [], []
with open(DPO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        pair = json.loads(line)
        if is_meaningfully_different(pair["chosen"], pair["rejected"]):
            good_pairs.append(pair)
        else:
            bad_pairs.append(pair)

print(f"Good pairs: {len(good_pairs)}")
print(f"Bad pairs (need regeneration): {len(bad_pairs)}")

missing_qid = sum(1 for p in bad_pairs if p.get("_qid") not in raft_by_qid)
print(f"Bad pairs missing a RAFT match: {missing_qid}")

# ------------------------------------------------------------------ #
# 3. Try cheap entity-swap first, collect the rest for LLM
# ------------------------------------------------------------------ #
fixed_pairs = []
needs_llm = []          # list of (pair, rec)

for pair in bad_pairs:
    rec = raft_by_qid.get(pair.get("_qid"))
    if rec is None:
        continue

    swapped = corrupt_answer(pair["chosen"])
    if swapped and is_meaningfully_different(pair["chosen"], swapped):
        pair["rejected"] = swapped
        fixed_pairs.append(pair)
    else:
        needs_llm.append((pair, rec))

print(f"\nFixed via entity-swap (CPU): {len(fixed_pairs)}")
print(f"Needs LLM fallback: {len(needs_llm)}")

# ------------------------------------------------------------------ #
# 4. LLM regeneration for the remaining hard cases
# ------------------------------------------------------------------ #
if needs_llm:
    print("\nRunning LLM regeneration on dual GPUs...")
    t0 = time.time()
    prompts = [build_rejected_gen_prompt_v2(rec, pair["chosen"]) for pair, rec in needs_llm]
    rejected_v2 = batch_generate_rejected_v2(prompts)

    still_bad = 0
    for (pair, rec), new_rejected in zip(needs_llm, rejected_v2):
        if new_rejected and is_meaningfully_different(pair["chosen"], new_rejected):
            pair["rejected"] = new_rejected
            fixed_pairs.append(pair)
        else:
            still_bad += 1

    print(f"Fixed via LLM: {len(needs_llm) - still_bad}")
    print(f"Still bad after retry (dropped): {still_bad}")
    print(f"LLM stage took {time.time() - t0:.1f}s")

# ------------------------------------------------------------------ #
# 5. Write fixed dataset
# ------------------------------------------------------------------ #
final_pairs = good_pairs + fixed_pairs
with open(FIXED_FILE, "w", encoding="utf-8") as f:
    for p in final_pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n🎉 Intermediate fixed dataset: {len(final_pairs)} pairs")
print(f"  - Originally good : {len(good_pairs)}")
print(f"  - Fixed           : {len(fixed_pairs)}")
print(f"  - Dropped         : {len(bad_pairs) - len(fixed_pairs)}")
print(f"Saved to: {FIXED_FILE}")

# ------------------------------------------------------------------ #
# 6. Final clean version (remove _qid and any extra keys)
# ------------------------------------------------------------------ #
with open(FIXED_FILE, "r", encoding="utf-8") as fin, \
     open(FINAL_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        p = json.loads(line)
        clean = {
            "prompt":   p["prompt"],
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }
        fout.write(json.dumps(clean, ensure_ascii=False) + "\n")

print(f"\nClean final file saved to: {FINAL_FILE}")
print("Done.")