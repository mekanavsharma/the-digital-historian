# phase_4_rlhf_faithfulness/config.py

import os

BASE_DIR = os.getcwd()

# --- Upstream artefacts (Phase 3) ---
from phase_3_moe_raft.config import RAFT_MODEL_PATH, BASE_LLM_MODEL_PATH, TOP_K  # noqa: E402

SFT_MODEL_PATH = RAFT_MODEL_PATH          # policy we start DPO from (Phase 3 output)
BASE_MODEL_PATH = BASE_LLM_MODEL_PATH

# --- Phase 4 paths ---
PHASE4_DIR = os.path.join(BASE_DIR, "phase_4_rlhf_faithfulness")
DELIVERABLE_DIR = os.path.join(PHASE4_DIR, "rlhf_faithfulness")

PREFERENCE_DATA_PATH = os.path.join(PHASE4_DIR, "data", "dpo_preference.jsonl")
DPO_OUTPUT_DIR = os.path.join(PHASE4_DIR, "dpo_finetuned")

RAGAS_RESULTS_CSV = os.path.join(DELIVERABLE_DIR, "ragas_metrics.csv")
HALLUCINATION_EXAMPLES_MD = os.path.join(DELIVERABLE_DIR, "before_after_examples.md")
METRICS_TABLE_MD = os.path.join(DELIVERABLE_DIR, "metrics_table.md")

# --- Preference-data generation ---
CANDIDATES_PER_PROMPT = 4          # how many samples to draw per (query, docs) pair
SAMPLING_TEMPERATURE = 0.8
SAMPLING_TOP_P = 0.9
MAX_NEW_TOKENS = 256

# Reward-shaping weights used to rank candidates (see citation_utils.score_candidate)
W_CITATION_VALIDITY = 0.4   # cites real, retrieved doc indices
W_FAITHFULNESS = 0.5        # cited claims are actually supported by cited doc text
W_ABSTENTION_BONUS = 0.1    # correctly says "I don't know" when nothing supports the answer

# --- Self-correction (STaR / Reflection) loop ---
MAX_REFINE_ROUNDS = 2
FAITHFULNESS_ACCEPT_THRESHOLD = 0.75   # if critique score >= this, stop refining

# --- DPO training hyperparameters ---
DPO_LORA_R = 16
DPO_LORA_ALPHA = 32
DPO_LORA_DROPOUT = 0.05
DPO_BETA = 0.1                 # DPO temperature
DPO_LEARNING_RATE = 5e-5
DPO_BATCH_SIZE = 1
DPO_GRAD_ACCUM = 4
DPO_NUM_EPOCHS = 2
DPO_MAX_LENGTH = 2048
DPO_MAX_PROMPT_LENGTH = 1536
