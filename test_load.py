# test_load.py — run once to confirm everything works
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./lfm-instruct-dynamic")

model = AutoModelForCausalLM.from_pretrained(
    "LiquidAI/LFM2.5-1.2B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "Say hello in exactly one sentence."}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    tokenize=True,
).to(model.device)

with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=60, do_sample=False)

response = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"\nResponse: {response}")
print("\n✅ Model works. Move to Step 6.")
