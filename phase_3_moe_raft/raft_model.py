import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from phase_3_moe_raft.config import RAFT_MODEL_PATH

def build_system_prompt(domain: str = None, perspective: str = None) -> str:
    base = "You are a knowledgeable historian. "
    if domain and perspective:
        base += f"Answer the question from a **{perspective} historian's viewpoint** focusing on the **{domain} period**. "
    elif perspective:
        base += f"Answer from a **{perspective}** perspective. "
    elif domain:
        base += f"Focus on the **{domain}** period. "
    base += (
        "Use ONLY the provided DOCUMENTS to answer. "
        "If the documents contain relevant information, synthesise a clear answer. "
        "If absolutely no relevant information is present, respond with 'I don't know'. "
        "Be concise but thorough."
    )
    return base

class RAFTModel:
    def __init__(self, model_path=None, debug=False):
        if model_path is None:
            model_path = RAFT_MODEL_PATH
        self.debug = debug

        # 4‑bit config for memory efficiency (optional – keeps model fast and small)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Detect if model_path is a PEFT adapter directory
        is_adapter = (
            os.path.isdir(model_path) and
            os.path.isfile(os.path.join(model_path, "adapter_config.json"))
        )

        if is_adapter:
            # Read base model name from adapter config
            with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                adapter_conf = json.load(f)
            base_model_name = adapter_conf.get("base_model_name_or_path")
            if base_model_name is None:
                raise ValueError("adapter_config.json does not contain 'base_model_name_or_path'")

            # Load the base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            # Load the tokenizer from the adapter directory (or base model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Attach the LoRA adapter
            self.model = PeftModel.from_pretrained(self.model, model_path)
            print(f"Loaded RAFT fine‑tuned model (base: {base_model_name})")

        else:
            # Load a regular model (or a fully merged model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"Loaded model from {model_path}")

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endfor %}"
            )

    def answer(self, documents: str, question: str,
               domain: str = None, perspective: str = None) -> str:
        system_prompt = build_system_prompt(domain, perspective)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"DOCUMENTS:\n{documents}\n\nQUESTION: {question}"}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.debug:
            print("\n--- DEBUG PROMPT ---")
            print(prompt)
            print("--------------------\n")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant\n" in full:
            answer = full.split("assistant\n")[-1].strip()
        else:
            answer = full[len(prompt):].strip()
        return answer