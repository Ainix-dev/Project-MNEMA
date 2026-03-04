# run_baseline.py
import sys
import os
sys.path.insert(0, '.')

from config import cfg
from model.loader import load_model_and_tokenizer, verify_base_frozen
from eval.baseline import save_baseline

model, tokenizer = load_model_and_tokenizer()
verify_base_frozen(model)
scores = save_baseline(model, tokenizer)
print(f"Baseline: {scores}")

os.makedirs(cfg.adapter_path, exist_ok=True)
model.save_pretrained(cfg.adapter_path)
tokenizer.save_pretrained(cfg.adapter_path)
print(f"Adapter saved to {cfg.adapter_path}")
