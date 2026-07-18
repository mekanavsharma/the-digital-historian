

# phase_3_moe_raft/train_raft.py
"""
RAFTAFT fine‑tuning with QLoRA – handles nested documents if present,
and uses modern SFTConfig to avoid deprecation warnings.
"""
import json, os, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from phase_3_moe_raft.config import (RAFT_TRAIN_JSON)

# ------------------------------------------------------------------ #
#  User‑editable configuration
# ------------------------------------------------------------------ #
BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # "Qwen/Qwen2.5-1.5B-Instruct"
RAFT_TRAIN_PATH = RAFT_TRAIN_JSON       # your JSONL file
OUTPUT_DIR = "phase_3_moe_raft/raft_finetuned"

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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install PyTorch with CUDA support.")

    # 4‑bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k‑bit training + LoRA
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_raft_jsonl(RAFT_TRAIN_PATH)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Use SFTConfig to avoid deprecation warnings
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
        max_seq_length=MAX_SEQ_LENGTH,   # this is now inside SFTConfig
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        formatting_func=formatting_func,
        args=sft_config,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training finished. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

