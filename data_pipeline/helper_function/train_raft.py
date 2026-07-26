"""
RAFTAFT fine‑tuning with QLoRA – handles nested documents if present,
and uses modern SFTConfig to avoid deprecation warnings.
"""
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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ------------------------------------------------------------------ #
#  User‑editable configuration
# ------------------------------------------------------------------ #
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"   # "Qwen/Qwen2.5-3B-Instruct"
RAFT_TRAIN_PATH = "/kaggle/input/datasets/kanav608/raft-dpo-data/raft_data.jsonl"
OUTPUT_DIR =  "/kaggle/working/raft_model/"

# QLoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]

# Training
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_SEQ_LENGTH = 2048
SAVE_STEPS = 200
LOGGING_STEPS = 20
# ------------------------------------------------------------------ #


def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def load_raft_jsonl(path):
    """Read JSONL and return a HuggingFace Dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return Dataset.from_list(examples)

def _to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_to_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def formatting_func(example):
    # Safely extract documents – handle nested list if present
    docs = example.get("documents", [])
    if isinstance(docs, list) and len(docs) > 0 and isinstance(docs[0], list):
        # Flatten one level
        docs = docs[0]

    if isinstance(docs, (list, tuple)):
        docs_text = "\n".join(_to_text(doc) for doc in docs)
    else:
        docs_text = _to_text(docs)

    system_msg = _to_text(example.get("instruction") or (
        "You are a helpful assistant. Use the provided documents to answer the question. "
        "If the answer cannot be found, say 'I don't know'."
    ))
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": f"{docs_text}\n\nQuestion: {_to_text(example.get('question', ''))}"},
        {"role": "assistant", "content": _to_text(example.get("output", ""))},
    ]

    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        # Fallback for tokenizers/templates that are stricter about content types
        rendered = ""
        for msg in messages:
            rendered += f"<|{msg['role']}|>\n{_to_text(msg['content'])}\n"

    # TRL expects a list of processed strings for each example.
    return [rendered]


def main():
    global tokenizer

    local_rank, rank, world_size = setup_distributed()
    is_main = rank == 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")

    if not os.path.exists(RAFT_TRAIN_PATH):
        raise FileNotFoundError(f"RAFT train dataset not found:\n{RAFT_TRAIN_PATH}")

    if is_main:
        print("=" * 70)
        print("DISTRIBUTED RAFT (SFT) TRAINING")
        print("=" * 70)
        print(f"World size: {world_size}")
        print(f"GPUs: {torch.cuda.device_count()}")
        print(f"Base model: {BASE_MODEL_NAME}")
        print()

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if is_main:
        print("Loading base model...")

    # Critical for DDP: pin each process to its own GPU
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training + LoRA
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        model.print_trainable_parameters()

    # Load dataset
    dataset = load_raft_jsonl(RAFT_TRAIN_PATH)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    if is_main:
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Eval examples:  {len(dataset['test'])}")

    # Use SFTConfig (modern TRL API)
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=3,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        # Distributed training
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_func,
        args=sft_config,
    )

    if is_main:
        print()
        print("=" * 70)
        print("STARTING RAFT TRAINING")
        print("=" * 70)
        print()

    trainer.train()

    if is_main:
        print()
        print("Saving RAFT model...")
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print()
        print("=" * 70)
        print("RAFT TRAINING FINISHED")
        print("=" * 70)
        print(f"Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


