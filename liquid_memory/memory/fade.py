# memory/fade.py
import math
import time
from memory.store import MemoryStore
from config import cfg


class EbbinghausDecay:
    """
    Implements the forgetting curve: strength(t) = S₀ * e^(-λ * Δt)
    
    But with important modifications:
    - λ is modulated by importance (important memories decay slower)
    - Repeated access resets the decay clock (spaced repetition)
    - Different memory types get different base rates
    """

    # Decay rate multipliers by type (lower = slower decay)
    TYPE_DECAY_MODIFIERS = {
        "correction":  0.3,   # very slow — corrections matter
        "preference":  0.5,   # slow — preferences are stable
        "fact":        0.8,   # medium
        "casual":      1.5,   # fast — casual chat fades quickly
    }

    def __init__(self, store: MemoryStore):
        self.store = store

    def run_decay_pass(self):
        """
        Apply the forgetting curve to all alive memories.
        Should be called every cfg.decay_interval_hours hours.
        """
        now = time.time()
        memories = self.store.get_all_for_decay()
        archived_count = 0

        for mem in memories:
            hours_since_reinforce = (now - mem["last_reinforced"]) / 3600.0

            # Effective λ: base rate * type modifier, reduced by importance
            type_modifier = self.TYPE_DECAY_MODIFIERS.get(
                # Need type — update get_all_for_decay to include it
                "fact", 1.0
            )
            effective_lambda = (
                cfg.decay_lambda
                * type_modifier
                * (1.0 - 0.5 * mem["importance"])  # high importance = slower decay
            )

            new_strength = mem["strength"] * math.exp(
                -effective_lambda * hours_since_reinforce
            )
            new_strength = max(0.0, new_strength)

            self.store.update_strength(mem["id"], new_strength)
            if new_strength < cfg.min_strength_threshold:
                archived_count += 1

        print(f"[Decay] Processed {len(memories)} memories, archived {archived_count}")
        return archived_count
