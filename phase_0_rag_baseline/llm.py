# phase_0_rag_baseline/llm.py
from typing import List, Optional, Tuple
import re
import torch
from transformers import BitsAndBytesConfig

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

def load_llm(model_name: str):
    """
    Load tokenizer and model. Example model_name: "HuggingFaceTB/SmolLM3-3B"
    Returns: (tokenizer, model)
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError("transformers not installed. pip install transformers accelerate -U")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Ensure pad_token exists (Qwen uses <|endoftext|> usually)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cuda",
        attn_implementation="sdpa", #  Use "sdpa" (Scaled Dot Product Attention) or "flash_attention_2" on modern GPU
    )

    model.eval()
    return tokenizer, model

def get_max_new_tokens(answer_style: str, max_words: Optional[int] = None) -> int:
    """Dynamically set max_new_tokens based on style to avoid truncation."""
    base = {
        "short": 200,
        "concise": 400,
        "detailed": 1200,
    }.get(answer_style, 400)
    if max_words:
        base = min(base, max_words * 2)  # Rough token estimate
    return base + 200  # Extra buffer for citations

def generate_answer(
    prompt: str,
    hf_tuple: Optional[Tuple],
    chunk_ids_used: Optional[List[str]] = None,
    answer_style: str = "concise",
    max_words: Optional[int] = None
    ):
    """
    Generates an answer using the HF LLM passed as hf_tuple = (tokenizer, model).
    Robustly removes prompt echo by slicing tokens, not strings.
    max_words = controls how many words will be generated, update it according to answer_style
    """
    if hf_tuple is None:
        # mock generation: return trimmed context for debugging retrieval/prompt
        try:
            context = prompt.split("CONTEXT:\n", 1)[1].split("\n\nQUESTION:", 1)[0]
            return context.strip()[:600]
        except Exception:
            return "I don't know."

    tokenizer, model = hf_tuple
    device = next(model.parameters()).device

    # Extract all available chunk_ids from the prompt
    available_chunks = re.findall(r'\[chunk_id=([^\],]+)', prompt)

    # System message emphasizing citations
    messages = [
        {
            "role": "system",
            "content": "You are a strictly grounded history assistant. Answer using ONLY the provided context.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    input_len = inputs["input_ids"].shape[1]
    max_new_tokens = get_max_new_tokens(answer_style, max_words)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,   # Provide more diverse output
            temperature=0.3,  # Low but not zero - helps with citation variety
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_len:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Post-processing
    if decoded.lower().startswith("i don't know"):
        return "I don't know"

    # Always attach authoritative citations from chunk_ids_used
    if chunk_ids_used:
        unique_ids = []
        seen = set()
        for cid in chunk_ids_used:
            if cid and cid not in seen:
                seen.add(cid)
                unique_ids.append(cid)

        citation_line = " ".join(f"[chunk_id={cid}]" for cid in unique_ids)
        decoded = decoded.strip() + "\n" + "\n" + citation_line

    return decoded
