# main.py — MNEMA v2 Phase 4 (Complete)
from model.loader import load_model_and_tokenizer, verify_base_frozen
from memory.graph import RelationalMemoryGraph
from memory.extractor import MemoryExtractor
from memory.goals import GoalUtilityLayer
from memory.metacog import MetaCognition
from memory.asc import AdaptiveStateCore
from memory.hardware import HardwareMonitor
from model.inference import chat
from scheduler import MemoryScheduler

# ── Display settings ──────────────────────────────────────────────────────────
SHOW_THINKING    = True
SHOW_MEMORY_TAGS = True
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("\n" + "═" * 55)
    print("  MNEMA v2 — A mind that remembers")
    print("═" * 55 + "\n")

    model, tokenizer = load_model_and_tokenizer()
    verify_base_frozen(model)

    memory_graph = RelationalMemoryGraph()
    extractor    = MemoryExtractor()
    goal_layer   = GoalUtilityLayer()
    metacog      = MetaCognition()
    asc          = AdaptiveStateCore()
    hardware     = HardwareMonitor(check_every_n_turns=3)

    scheduler = MemoryScheduler(memory_graph, model, tokenizer)
    scheduler.start()

    conversation_history = []
    turn_counter = 0

    # Show initial hardware tier
    snap = hardware._last_snapshot
    if snap:
        print(f"  Hardware: {snap.tier} tier "
              f"({snap.vram_free_gb:.2f}GB VRAM free)\n")

    print("Commands:")
    print("  memory    → memory graph nodes")
    print("  graph     → graph stats + contradictions")
    print("  goals     → goal performance scores")
    print("  metacog   → self-modeling state")
    print("  asc       → adaptive behavioral state")
    print("  hw        → hardware status + active tier")
    print("  think on  → show inner monologue")
    print("  think off → hide inner monologue")
    print("  clear     → wipe memory")
    print("  quit      → exit\n")

    global SHOW_THINKING

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # ── Commands ──────────────────────────────────────────────────────
            if user_input.lower() == "quit":
                break

            if user_input.lower() == "memory":
                _show_memories(memory_graph)
                continue

            if user_input.lower() == "graph":
                _show_graph_stats(memory_graph)
                continue

            if user_input.lower() == "goals":
                _show_goals(goal_layer)
                continue

            if user_input.lower() == "metacog":
                print(metacog.display_summary())
                continue

            if user_input.lower() == "asc":
                print(asc.display_state())
                continue

            if user_input.lower() == "hw":
                print(hardware.display_status())
                continue

            if user_input.lower() == "think on":
                SHOW_THINKING = True
                if not hardware.thinking_allowed(True):
                    print("  [Monologue requested but VRAM is too low — "
                          "will enable when headroom improves]\n")
                else:
                    print("  [Inner monologue visible]\n")
                continue

            if user_input.lower() == "think off":
                SHOW_THINKING = False
                print("  [Inner monologue hidden]\n")
                continue

            if user_input.lower() == "clear":
                _clear_memory()
                print("  [Memory cleared]\n")
                continue

            turn_counter += 1

            # ── Pause scheduler if hardware demands it ────────────────────────
            if hardware.should_pause_scheduler():
                scheduler.pause()
            else:
                scheduler.resume()

            # ── Extract and store memories ────────────────────────────────────
            new_memories = extractor.extract(user_input, turn_counter)
            for mem in new_memories:
                memory_graph.add(
                    content=mem["content"],
                    memory_type=mem["type"],
                    importance=mem["importance"],
                    turn=turn_counter
                )
                if SHOW_MEMORY_TAGS:
                    print(f"  \033[2m[memory: {mem['type']} · "
                          f"importance={mem['importance']:.1f}]\033[0m")

            # ── Generate response ──────────────────────────────────────────────
            spoken, monologue = chat(
                model, tokenizer, user_input,
                memory_graph, conversation_history,
                show_thinking=SHOW_THINKING,
                goal_layer=goal_layer,
                metacog=metacog,
                asc=asc,
                hardware=hardware
            )

            # ── Display ────────────────────────────────────────────────────────
            if SHOW_THINKING and monologue:
                print(f"\n  \033[2m\033[3m💭 {monologue}\033[0m\n")

            print(f"\nMNEMA: {spoken}\n")

            # ── Update history ─────────────────────────────────────────────────
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": spoken})

            if len(conversation_history) > 2000:
                conversation_history = conversation_history[-2000:]

    except KeyboardInterrupt:
        print("\n\n  Goodbye.")
    finally:
        asc._save_state(turn_counter)
        scheduler.stop()


def _show_memories(graph: RelationalMemoryGraph):
    memories = graph.get_all_for_decay()
    if not memories:
        print("\n  [No memories stored yet]\n")
        return
    print(f"\n  ── Memory Graph ({len(memories)} alive nodes) ──")
    for mem in sorted(memories, key=lambda x: x["strength"], reverse=True)[:10]:
        bar_len = int(mem["strength"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {bar} {mem['strength']:.2f} [{mem['type']}] "
              f"{mem['content'][:55]}...")
    print()


def _show_graph_stats(graph: RelationalMemoryGraph):
    stats = graph.stats()
    print(f"\n  ── Graph Stats ──")
    print(f"  Alive nodes:       {stats['alive_nodes']}")
    print(f"  Superseded nodes:  {stats['superseded_nodes']}")
    print(f"  Total edges:       {stats['total_edges']}")
    print(f"  Contradictions:    {stats['contradictions']}")
    if stats['edge_breakdown']:
        print(f"  Edge breakdown:")
        for edge_type, count in stats['edge_breakdown'].items():
            print(f"    {edge_type:<15} {count}")
    contradictions = graph.get_contradictions()
    if contradictions:
        print(f"\n  ── Resolved Contradictions ──")
        for c in contradictions[:5]:
            print(f"  SUPERSEDED: {c['older_content'][:55]}...")
            print(f"  CURRENT:    {c['newer_content'][:55]}...")
            print(f"  confidence: {c['confidence']:.2f}\n")
    print()


def _show_goals(goal_layer: GoalUtilityLayer):
    summary = goal_layer.get_goal_summary()
    print(f"\n  ── Goal Performance ──")
    for goal_id, data in summary.items():
        bar_len = int(data["score"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {bar} {data['score']:.2f} {goal_id} ({data['trend']})")
        print(f"    {data['description']}")
    print(f"\n  Strongest:  {goal_layer.get_strongest_goal()}")
    print(f"  Needs work: {goal_layer.get_weakest_goal()}\n")


def _clear_memory():
    import os, shutil
    paths = ["./data/memory_graph.db", "./data/chroma", "./data/memory.db"]
    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)


if __name__ == "__main__":
    main()
