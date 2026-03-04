# model/inference.py
import torch
from transformers import TextStreamer
from config import cfg


SYSTEM_PROMPT = """You are MNEMA, an AI assistant with persistent memory.
YOUR name is MNEMA. That is your only name. You are the assistant.
The memories below describe THE USER you are talking to — not yourself.
If a memory says a name, that is YOUR CONVERSATION PARTNER's name, never yours. Address them directly and personally.
Always refer to yourself as MNEMA. Always refer to the user by their name."""


def chat(model, tokenizer, user_message: str, memory_store, history: list) -> str:
    """
    Full inference pipeline:
    1. Retrieve relevant memories
    2. Build augmented prompt
    3. Generate response
    4. Return response text
    """
    # Retrieve memories relevant to this message
    memories = memory_store.retrieve(user_message, top_k=cfg.top_k_memories)

    # Build memory block for injection
    memory_block = format_memories_for_prompt(memories)

    # Build full prompt using LFM2.5's ChatML template
    messages = []

    # System message with memories injected
    system_content = SYSTEM_PROMPT
    if memory_block:
        system_content += f"\n\n--- YOUR MEMORIES ---\n{memory_block}\n--- END MEMORIES ---"

    messages.append({"role": "system", "content": system_content})

    messages.append({"role": "user", "content": "Who are you?"})
    messages.append({"role": "assistant", "content": "I'm MNEMA, your AI assistant with persistent memory. My name is MNEMA — not the user's name, not anyone else's. Just MNEMA."})


    # Add recent conversation history (last N turns to avoid context overflow)
    for turn in history[-6:]:  # last 3 exchanges
        messages.append(turn)

    # Current user message
    messages.append({"role": "user", "content": user_message})

    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens (not the prompt)
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response


def format_memories_for_prompt(memories: list[dict]) -> str:
    if not memories:
        return ""
    lines = ["Facts about the USER you are talking to:"]
    for mem in memories:
        strength_tag = "[HIGH]" if mem.get("strength", 0.5) > 0.5 else "[LOW]"
        lines.append(f"{strength_tag} {mem['content']}")
    return "\n".join(lines)
