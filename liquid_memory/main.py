# main.py
from model.loader import load_model_and_tokenizer, verify_base_frozen
from memory.store import MemoryStore
from memory.extractor import MemoryExtractor
from model.inference import chat
from scheduler import MemoryScheduler


def main():
    print("=== LFM2.5 Continuous Learning System ===")

    # Load everything
    model, tokenizer = load_model_and_tokenizer()
    verify_base_frozen(model)  # safety check at startup

    memory_store = MemoryStore()
    extractor = MemoryExtractor()

    # Start background decay + consolidation
    scheduler = MemoryScheduler(memory_store, model, tokenizer)
    scheduler.start()

    conversation_history = []
    turn_counter = 0

    print("Chat started. Type 'quit' to exit, 'memory' to see stored memories.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "memory":
                _show_memories(memory_store)
                continue

            turn_counter += 1

            # === STEP 1: Extract memories from user's message ===
            new_memories = extractor.extract(user_input, turn_counter)
            for mem in new_memories:
                memory_store.add(
                    content=mem["content"],
                    memory_type=mem["type"],
                    importance=mem["importance"],
                    turn=turn_counter
                )
                print(f"  [Memory stored: {mem['type']} / importance={mem['importance']:.1f}]")

            # === STEP 2: Generate response with memory-augmented context ===
            response = chat(model, tokenizer, user_input, memory_store, conversation_history)
            print(f"\nAssistant: {response}\n")

            # === STEP 3: Update conversation history ===
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response})

            # Keep history bounded
            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        scheduler.stop()
        print("Scheduler stopped. Goodbye.")


def _show_memories(store):
    """Debug helper to inspect memory state."""
    memories = store.get_all_for_decay()
    print(f"\n--- Memory State ({len(memories)} alive) ---")
    for mem in sorted(memories, key=lambda x: x["strength"], reverse=True)[:10]:
        print(f"  strength={mem['strength']:.3f} | {mem.get('content', '')[:80]}...")
    print("---\n")


if __name__ == "__main__":
    main()
