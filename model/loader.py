# model/loader.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig, get_peft_model, PeftModel,
    TaskType, prepare_model_for_kbit_training
)
from config import cfg


def load_model_and_tokenizer():
    """
    Loads LFM2.5 2B with weights FROZEN.
    Adds a LoRA adapter if one exists, otherwise creates a fresh one.
    The base model weights are NEVER modified.
    """
    print(f"Loading tokenizer from {cfg.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load base model (FROZEN) ---
    quant_config = None
    if cfg.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    print("Loading base model (frozen)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        quantization_config=quant_config,
        dtype=torch.bfloat16 if not cfg.load_in_4bit else None,
        device_map="auto",
        trust_remote_code=True,  # required for LFM2.5
    )

    # CRITICAL: Freeze ALL base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    if cfg.load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    # --- Attach LoRA adapter ---
    if os.path.exists(cfg.adapter_path):
        print(f"Loading existing LoRA adapter from {cfg.adapter_path}...")
        model = PeftModel.from_pretrained(
            base_model,
            cfg.adapter_path,
            is_trainable=True,          # allow future consolidation updates
        )
    else:
        print("Creating fresh LoRA adapter...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
        )
        model = get_peft_model(base_model, lora_config)

    # Verify: count trainable vs frozen
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    # You should see ~0.5-1% trainable — if it's 100%, something went wrong

    model.eval()  # start in eval mode
    return model, tokenizer


def verify_base_frozen(model):
    """Safety check — call this anytime to verify base weights are still frozen."""
    for name, param in model.named_parameters():
        if "lora_" not in name and "modules_to_save" not in name:
            if param.requires_grad:
                raise RuntimeError(
                    f"BASE MODEL WEIGHT IS TRAINABLE: {name}\n"
                    "This should never happen. Check your loader."
                )
    print("✓ Base model weights confirmed frozen.")
