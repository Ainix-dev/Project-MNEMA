# memory/goals.py
"""
MNEMA v2 — Goal & Utility Layer

Gives MNEMA purpose-driven memory retrieval and interaction scoring.
Instead of retrieving memories purely by semantic similarity, memories
are also ranked by how relevant they are to active goals.

Built-in goals:
    minimize_corrections   — penalize turns where user corrects MNEMA
    match_tone             — reward turns where user engages positively
    be_concise             — reward when user doesn't ask to elaborate
    remember_context       — reward when MNEMA uses memory correctly
    build_trust            — long-term relationship quality score

Each goal has:
    weight      — how much it influences memory retrieval priority
    score       — running performance score (0.0 - 1.0)
    history     — list of recent outcomes for trend analysis

Every interaction turn:
    1. Signals are detected from user input (correction, positive, etc.)
    2. Goals are scored based on signals
    3. Memories are tagged with utility scores linking them to goals
    4. Retrieval blends semantic similarity + utility score
"""

import sqlite3
import time
import json
import math
from dataclasses import dataclass, field


# ── Goal definitions ─────────────────────────────────────────────────────────

DEFAULT_GOALS = {
    "minimize_corrections": {
        "weight":      0.30,
        "score":       1.0,       # starts optimistic
        "description": "Avoid being corrected by the user",
        "history":     [],
    },
    "match_tone": {
        "weight":      0.25,
        "score":       0.8,
        "description": "Match the user's conversational energy and style",
        "history":     [],
    },
    "be_concise": {
        "weight":      0.15,
        "score":       0.8,
        "description": "Be appropriately brief — don't over-explain",
        "history":     [],
    },
    "remember_context": {
        "weight":      0.20,
        "score":       0.9,
        "description": "Use stored memories correctly in responses",
        "history":     [],
    },
    "build_trust": {
        "weight":      0.10,
        "score":       0.8,
        "description": "Long-term relationship quality and consistency",
        "history":     [],
    },
}

# ── Signal detection patterns ─────────────────────────────────────────────────

CORRECTION_SIGNALS = [
    "no,", "wrong", "incorrect", "not right", "actually",
    "i said", "i meant", "that's not", "you're wrong",
    "you misunderstood", "that's wrong", "not what i"
]

POSITIVE_SIGNALS = [
    "exactly", "yes", "correct", "right", "perfect",
    "that's it", "good", "great", "thanks", "thank you",
    "you remembered", "you know me", "that's right"
]

ELABORATION_SIGNALS = [
    "can you explain", "what do you mean", "elaborate",
    "tell me more", "go on", "expand on", "more detail"
]

MEMORY_USE_SIGNALS = [
    "you remember", "you know", "as i said", "like i told you",
    "from before", "last time", "earlier", "you mentioned"
]


@dataclass
class TurnSignals:
    """Signals detected from a single conversation turn."""
    correction: bool = False
    positive: bool = False
    asked_to_elaborate: bool = False
    memory_acknowledged: bool = False
    raw_text: str = ""


class GoalUtilityLayer:
    """
    Tracks MNEMA's goals and scores memories by utility.
    Persists goal state to SQLite so scores survive across sessions.
    """

    def __init__(self, db_path: str = "./data/memory_graph.db"):
        self.db_path = db_path
        self.goals = dict(DEFAULT_GOALS)
        self._init_db()
        self._load_goals()

    # ── Database ─────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS goals (
                id          TEXT PRIMARY KEY,
                weight      REAL DEFAULT 0.2,
                score       REAL DEFAULT 0.8,
                description TEXT,
                history     TEXT DEFAULT '[]',
                updated_at  REAL
            );

            CREATE TABLE IF NOT EXISTS node_utility (
                node_id     TEXT NOT NULL,
                goal_id     TEXT NOT NULL,
                utility     REAL DEFAULT 0.5,
                updated_at  REAL,
                PRIMARY KEY (node_id, goal_id)
            );
        """)
        conn.commit()
        conn.close()

    def _load_goals(self):
        """Load persisted goal scores from DB, fall back to defaults."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT id, weight, score, description, history FROM goals"
        ).fetchall()
        conn.close()

        for row in rows:
            goal_id = row[0]
            if goal_id in self.goals:
                self.goals[goal_id]["weight"] = row[1]
                self.goals[goal_id]["score"] = row[2]
                try:
                    self.goals[goal_id]["history"] = json.loads(row[4])
                except (json.JSONDecodeError, TypeError):
                    self.goals[goal_id]["history"] = []

        # Initialize any goals not yet in DB
        self._save_goals()

    def _save_goals(self):
        conn = sqlite3.connect(self.db_path)
        for goal_id, goal in self.goals.items():
            conn.execute("""
                INSERT OR REPLACE INTO goals
                (id, weight, score, description, history, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (goal_id, goal["weight"], goal["score"],
                  goal["description"],
                  json.dumps(goal["history"][-50:]),  # keep last 50
                  time.time()))
        conn.commit()
        conn.close()

    # ── Signal detection ─────────────────────────────────────────────────────

    def detect_signals(self, user_message: str) -> TurnSignals:
        """Detect goal-relevant signals from a user message."""
        text = user_message.lower()
        signals = TurnSignals(raw_text=user_message)

        signals.correction = any(s in text for s in CORRECTION_SIGNALS)
        signals.positive = any(s in text for s in POSITIVE_SIGNALS)
        signals.asked_to_elaborate = any(s in text for s in ELABORATION_SIGNALS)
        signals.memory_acknowledged = any(s in text for s in MEMORY_USE_SIGNALS)

        return signals

    # ── Goal scoring ─────────────────────────────────────────────────────────

    def score_turn(self, signals: TurnSignals) -> dict:
        """
        Update goal scores based on signals from this turn.
        Returns dict of {goal_id: delta} for logging.
        """
        deltas = {}

        # minimize_corrections
        if signals.correction:
            delta = -0.15
        elif signals.positive:
            delta = +0.05
        else:
            delta = +0.01  # neutral turns slightly positive
        deltas["minimize_corrections"] = delta
        self._update_goal_score("minimize_corrections", delta, signals.correction)

        # match_tone
        if signals.positive:
            delta = +0.08
        elif signals.correction:
            delta = -0.05
        else:
            delta = +0.01
        deltas["match_tone"] = delta
        self._update_goal_score("match_tone", delta)

        # be_concise — penalize if user asks to elaborate
        if signals.asked_to_elaborate:
            delta = -0.10
        else:
            delta = +0.02
        deltas["be_concise"] = delta
        self._update_goal_score("be_concise", delta)

        # remember_context — reward if user acknowledges memory use
        if signals.memory_acknowledged:
            delta = +0.10
        elif signals.correction:
            delta = -0.08
        else:
            delta = +0.01
        deltas["remember_context"] = delta
        self._update_goal_score("remember_context", delta)

        # build_trust — slow-moving composite
        trust_delta = (
            (+0.03 if signals.positive else 0) +
            (-0.05 if signals.correction else 0) +
            (+0.01)  # baseline positive drift
        )
        deltas["build_trust"] = trust_delta
        self._update_goal_score("build_trust", trust_delta)

        self._save_goals()
        return deltas

    def _update_goal_score(self, goal_id: str, delta: float,
                            is_failure: bool = False):
        """Apply delta to goal score with momentum smoothing."""
        goal = self.goals[goal_id]
        # Exponential moving average — recent outcomes weighted more
        new_score = goal["score"] + delta
        new_score = max(0.0, min(1.0, new_score))
        goal["score"] = new_score
        goal["history"].append({
            "delta": delta,
            "score": new_score,
            "failure": is_failure,
            "ts": time.time()
        })

    # ── Utility scoring ───────────────────────────────────────────────────────

    def compute_utility(self, memory: dict, signals: TurnSignals) -> float:
        """
        Compute a utility score for a memory given current turn signals.
        Utility = weighted sum of goal relevance scores.

        Higher utility → injected higher in the prompt context.
        """
        mem_type = memory.get("type", "fact")
        utility = 0.0

        # Corrections are always high utility for minimize_corrections goal
        if mem_type == "correction":
            utility += self.goals["minimize_corrections"]["weight"] * 1.0

        # Preferences are high utility for match_tone and be_concise
        if mem_type == "preference":
            utility += self.goals["match_tone"]["weight"] * 0.9
            utility += self.goals["be_concise"]["weight"] * 0.7

        # Facts are high utility for remember_context
        if mem_type in ("fact", "event"):
            utility += self.goals["remember_context"]["weight"] * 0.8

        # Boost utility if current signals suggest this memory type is needed
        if signals.correction and mem_type == "correction":
            utility *= 1.5   # corrections during correction events = critical
        if signals.memory_acknowledged and mem_type in ("fact", "preference"):
            utility *= 1.3

        # Scale by memory strength — fading memories contribute less
        utility *= memory.get("strength", 1.0)

        # Scale by overall goal health — poor goals need better memories
        avg_goal_score = sum(
            g["score"] for g in self.goals.values()
        ) / len(self.goals)
        if avg_goal_score < 0.6:
            utility *= 1.2   # underperforming → boost all memory utility

        return min(1.0, utility)

    def tag_memories_with_utility(self, memories: list[dict],
                                   signals: TurnSignals) -> list[dict]:
        """
        Add utility scores to retrieved memories and re-rank them.
        Blends semantic similarity (hop/strength) with goal utility.
        """
        conn = sqlite3.connect(self.db_path)

        for mem in memories:
            utility = self.compute_utility(mem, signals)
            mem["utility"] = utility

            # Persist utility score
            conn.execute("""
                INSERT OR REPLACE INTO node_utility
                (node_id, goal_id, utility, updated_at)
                VALUES (?, ?, ?, ?)
            """, (mem["id"], "composite", utility, time.time()))

        conn.commit()
        conn.close()

        # Re-rank: blend hop distance, strength, and utility
        for mem in memories:
            hop_penalty = mem.get("hop", 0) * 0.2
            mem["_rank_score"] = (
                mem.get("strength", 0.5) * 0.4 +
                mem.get("utility", 0.5) * 0.4 +
                (1.0 - hop_penalty) * 0.2
            )

        memories.sort(key=lambda x: x["_rank_score"], reverse=True)
        return memories

    # ── Goal state inspection ─────────────────────────────────────────────────

    def get_goal_summary(self) -> dict:
        """Return current goal scores and trends for display."""
        summary = {}
        for goal_id, goal in self.goals.items():
            history = goal["history"][-10:]
            trend = "stable"
            if len(history) >= 3:
                recent = sum(h["delta"] for h in history[-3:])
                if recent > 0.05:
                    trend = "improving ↑"
                elif recent < -0.05:
                    trend = "declining ↓"
            summary[goal_id] = {
                "score": round(goal["score"], 3),
                "weight": goal["weight"],
                "trend": trend,
                "description": goal["description"],
            }
        return summary

    def get_weakest_goal(self) -> str:
        """Return the goal MNEMA is currently performing worst on."""
        return min(self.goals.items(), key=lambda x: x[1]["score"])[0]

    def get_strongest_goal(self) -> str:
        """Return the goal MNEMA is currently performing best on."""
        return max(self.goals.items(), key=lambda x: x[1]["score"])[0]
