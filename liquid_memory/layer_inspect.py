# layer_inspect.py — run once, then delete
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print("\n=== LINEAR LAYERS (LoRA candidates) ===")
import torch.nn as nn
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"  {name}")
