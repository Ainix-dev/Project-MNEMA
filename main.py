# main.py
import sys
from model.loader import load_model_and_tokenizer, verify_base_frozen
from memory.store import MemoryStore
from memory.extractor import MemoryExtractor
from model.inference import chat
from scheduler import MemoryScheduler

# ── Display settings ──────────────────────────────────────────
SHOW_THINKING = True      # set False to hide MNEMA's inner monologue
SHOW_MEMORY_TAGS = True   # set False to hide [Memory stored: ...] lines
# ─────────────────────────────────────────────────────────────


def main():
    print("\n" + "═" * 55)
    print("  MNEMA — A mind that remembers")
    print("═" * 55 + "\n")

    model, tokenizer = load_model_and_tokenizer()
    verify_base_frozen(model)

    memory_store = MemoryStore()
    extractor = MemoryExtractor()

    scheduler = MemoryScheduler(memory_store, model, tokenizer)
    scheduler.start()

    conversation_history = []
    turn_counter = 0

    print("Type 'quit' to exit")
    print("Type 'memory' to inspect stored memories")
    print("Type 'think on' / 'think off' to toggle inner monologue")
    print("Type 'clear' to wipe memory and start fresh\n")

    global SHOW_THINKING

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # ── Commands ──
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "memory":
                _show_memories(memory_store)
                continue
            if user_input.lower() == "think on":
                SHOW_THINKING = True
                print("  [Inner monologue visible]\n")
                continue
            if user_input.lower() == "think off":
                SHOW_THINKING = False
                print("  [Inner monologue hidden]\n")
                continue
            if user_input.lower() == "clear":
                _clear_memory()
                print("  [Memory cleared. Starting fresh.]\n")
                continue

            turn_counter += 1

            # ── Extract and store memories ──
            new_memories = extractor.extract(user_input, turn_counter)
            for mem in new_memories:
                memory_store.add(
                    content=mem["content"],
                    memory_type=mem["type"],
                    importance=mem["importance"],
                    turn=turn_counter
                )
                if SHOW_MEMORY_TAGS:
                    print(f"  \033[2m[memory: {mem['type']} · importance={mem['importance']:.1f}]\033[0m")

            # ── Generate response ──
            spoken, monologue = chat(
                model, tokenizer, user_input,
                memory_store, conversation_history,
                show_thinking=SHOW_THINKING
            )

            # ── Display inner monologue if enabled ──
            if SHOW_THINKING and monologue:
                print(f"\n  \033[2m\033[3m💭 {monologue}\033[0m\n")

            # ── Display spoken response ──
            print(f"\nMNEMA: {spoken}\n")

            # ── Update history ──
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": spoken})

            # Cap raw history in RAM — actual token trimming happens at inference
            if len(conversation_history) > 2000:
                conversation_history = conversation_history[-2000:]

    except KeyboardInterrupt:
        print("\n\n  Goodbye.")
    finally:
        scheduler.stop()


def _show_memories(store):
    memories = store.get_all_for_decay()
    if not memories:
        print("\n  [No memories stored yet]\n")
        return
    print(f"\n  ── Memory State ({len(memories)} alive) ──")
    for mem in sorted(memories, key=lambda x: x["strength"], reverse=True)[:10]:
        bar_len = int(mem["strength"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {bar} {mem['strength']:.2f}")
    print()


def _clear_memory():
    import os, shutil
    paths = ["./data/chroma", "./data/memory.db"]
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
