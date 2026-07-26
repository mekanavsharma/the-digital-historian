import os
import json

# IMPORTANT: set before importing torch/transformers
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

from phase_4_rlhf_faithfulness.config import (
    PREFERENCE_DATA_PATH, DPO_OUTPUT_DIR, DPO_BETA, DPO_BATCH_SIZE,
    DPO_GRAD_ACCUM, DPO_NUM_EPOCHS, DPO_MAX_LENGTH, DPO_MAX_PROMPT_LENGTH,
)

RAFT_MODEL_PATH = "/kaggle/working/raft_model/"
PREFERENCE_DATA_PATH = "/kaggle/input/datasets/kanav608/raft-dpo-data/preference_pairs_final.jsonl"
DPO_OUTPUT_DIR =  "/kaggle/working/dpo_model/"

MAX_PAIRS = 4000

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def load_preference_dataset(path: str) -> Dataset:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    ds = Dataset.from_list(rows)
    keep = {"prompt", "chosen", "rejected"}
    remove = [ c for c in ds.column_names if c not in keep]
    if remove: ds = ds.remove_columns(remove)
    return ds

def main():
    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not os.path.exists(PREFERENCE_DATA_PATH):
        raise FileNotFoundError( f"Preference dataset not found:\n" f"{PREFERENCE_DATA_PATH}" )
    if not os.path.exists(RAFT_MODEL_PATH):
        raise FileNotFoundError( f"RAFT adapter not found:\n" f"{RAFT_MODEL_PATH}" )

    if is_main:
        print("=" * 70)
        print("DISTRIBUTED DPO TRAINING")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"GPUs: {torch.cuda.device_count()}")
        print()

    adapter_config_path = os.path.join( RAFT_MODEL_PATH, "adapter_config.json" )
    if not os.path.isfile(adapter_config_path):
        raise FileNotFoundError(  f"No adapter_config.json found at:\n" f"{adapter_config_path}" )
    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config["base_model_name_or_path"]
    if is_main:
        print(f"Base model: {base_model_name}")
        print(f"RAFT adapter: {RAFT_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True,)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    if is_main:
        print("Loading base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        trust_remote_code=True,
    )

    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False

    policy_model = PeftModel.from_pretrained(
        base_model,
        RAFT_MODEL_PATH,
        is_trainable=True,
    )

    if is_main:
        print("Loaded Phase 3 RAFT adapter as DPO starting policy.")

    dataset = load_preference_dataset( PREFERENCE_DATA_PATH)

    if len(dataset) > MAX_PAIRS:
        dataset = (dataset.shuffle(seed=42).select(range(MAX_PAIRS)))

    if is_main:
        print(f"Total preference pairs: {len(dataset)}")

    dataset = dataset.train_test_split(test_size=0.05,seed=42)

    dpo_config = DPOConfig(
        output_dir=DPO_OUTPUT_DIR,
        beta=DPO_BETA,

        # Effective batch:
        # 2 × 4 = 8
        per_device_train_batch_size=DPO_BATCH_SIZE,
        per_device_eval_batch_size=DPO_BATCH_SIZE,
        gradient_accumulation_steps=DPO_GRAD_ACCUM,
        num_train_epochs=DPO_NUM_EPOCHS,
        learning_rate=5e-5,
        max_length=DPO_MAX_LENGTH,
        max_prompt_length=DPO_MAX_PROMPT_LENGTH,

        # We want to avoid frequent expensive evaluations
        eval_strategy="no",
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,

        fp16=True,
        bf16=False,

        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={ "use_reentrant": False },

        # Keep this because it solved OOM issues on Kaggle (even though it's not strictly necessary for DPO)
        remove_unused_columns=False,

        # Distributed training
        ddp_find_unused_parameters=False,
    )

    trainer = DPOTrainer(
        model=policy_model,
        # None means reference is derived by disabling the RAFT LoRA adapter.
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
    )


    if is_main:
        print()
        print("=" * 70)
        print("STARTING DPO TRAINING")
        print("=" * 70)
        print()

    trainer.train()

    if is_main:
        print()
        print("Saving DPO model...")
        trainer.model.save_pretrained(DPO_OUTPUT_DIR)
        tokenizer.save_pretrained(DPO_OUTPUT_DIR)

        print()
        print("=" * 70)
        print("DPO TRAINING FINISHED")
        print("=" * 70)
        print(f"Saved to: {DPO_OUTPUT_DIR}")


if __name__ == "__main__":
    main()