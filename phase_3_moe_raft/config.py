# phase_3_moe_raft/train_raft.py

import os

# Number of documents to retrieve
TOP_K = 20

# --- Paths to Phase 0 artefacts (adjust these to your real locations) ---
BASE_DIR = os.getcwd()

RAW_JSON_DIR = os.path.join(BASE_DIR, "data_pipeline", "data_jsonl")
PHASE0_CHUNKS_META = os.path.join(BASE_DIR, "index", "chunks_meta.json")
PHASE0_BM25_INDEX   = os.path.join(BASE_DIR, "index", "bm25.pkl")
PHASE0_FAISS_INDEX  = os.path.join(BASE_DIR, "index", "faiss", "index.faiss")

RAFT_TRAIN_JSON = os.path.join(BASE_DIR, "data_pipeline", "raft_train.jsonl")


# Fine‑tuned RAFT model (after training)
# "Qwen/Qwen3-4B-Instruct-2507"  #"Qwen/Qwen2.5-1.5B-Instruct" Qwen/Qwen2.5-3B-Instruct

# For MoE (base model) – can be any size you want (3B, 4B, etc.)
BASE_LLM_MODEL_PATH =  "Qwen/Qwen3-4B-Instruct-2507"

# For RAFT – points to your fine‑tuned adapter folder (trained on 1.5B)
RAFT_MODEL_PATH = os.path.join(BASE_DIR, "phase_3_moe_raft", "raft_finetuned")

# Embedding model used to encode queries for FAISS (must match the index)
EMBED_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Expert definitions ---
# Each expert is a combination of a time domain and a historian perspective.
EXPERTS = {
    # Nationalist perspectives
    "ancient_nationalist":      {"expert_domain": "Ancient",  "historian_perspective": "Nationalist"},
    "medieval_nationalist":     {"expert_domain": "Medieval", "historian_perspective": "Nationalist"},
    "modern_nationalist":       {"expert_domain": "Modern",   "historian_perspective": "Nationalist"},
    # Marxist perspectives
    "ancient_marxist":          {"expert_domain": "Ancient",  "historian_perspective": "Marxist"},
    "medieval_marxist":         {"expert_domain": "Medieval", "historian_perspective": "Marxist"},
    "modern_marxist":           {"expert_domain": "Modern",   "historian_perspective": "Marxist"},
    # Neutral / source‑verification perspective
    "ancient_neutral":          {"expert_domain": "Ancient",  "historian_perspective": "Neutral"},
    "medieval_neutral":         {"expert_domain": "Medieval", "historian_perspective": "Neutral"},
    "modern_neutral":           {"expert_domain": "Modern",   "historian_perspective": "Neutral"},
}

# Default expert when no match is found
DEFAULT_EXPERT = "modern_nationalist"

# Number of documents to retrieve
TOP_K = 15