# eval/baseline.py
"""
Run this ONCE at startup to capture baseline performance.
Compare against after every consolidation to detect degradation.
"""
import torch
import json

EVAL_PROMPTS = [
    # Reasoning
    {"prompt": "What is 17 × 24?", "check": "408"},
    {"prompt": "If all roses are flowers and some flowers fade, can some roses fade?", "check": "yes"},
    # Knowledge
    {"prompt": "What is the capital of Japan?", "check": "tokyo"},
    {"prompt": "Write a Python function to reverse a string.", "check": "def"},
    # Instruction following
    {"prompt": "List exactly 3 colors of the rainbow.", "check": lambda r: len(r.split('\n')) >= 3},
]


def run_eval(model, tokenizer) -> dict:
    scores = {}
    for i, item in enumerate(EVAL_PROMPTS):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response = tokenizer.decode(out[0], skip_special_tokens=True).lower()

        if callable(item["check"]):
            passed = item["check"](response)
        else:
            passed = item["check"].lower() in response

        scores[f"eval_{i}"] = 1 if passed else 0

    overall = sum(scores.values()) / len(scores)
    scores["overall"] = overall
    return scores


def save_baseline(model, tokenizer, path="./data/baseline_eval.json"):
    scores = run_eval(model, tokenizer)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Baseline saved: {scores}")
    return scores


def check_degradation(model, tokenizer, baseline_path="./data/baseline_eval.json", threshold=0.1):
    """Call after every consolidation. Alerts if performance drops."""
    import json
    with open(baseline_path) as f:
        baseline = json.load(f)
    current = run_eval(model, tokenizer)
    drop = baseline["overall"] - current["overall"]
    if drop > threshold:
        print(f"⚠️  DEGRADATION DETECTED: {drop:.2%} drop from baseline!")
        print(f"   Baseline: {baseline['overall']:.2%} → Current: {current['overall']:.2%}")
        print("   Consider rolling back to adapter backup.")
        return False
    print(f"✓ Model health check passed. Performance: {current['overall']:.2%} (baseline: {baseline['overall']:.2%})")
    return True
