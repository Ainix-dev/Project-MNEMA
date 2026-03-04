# memory/asc.py
"""
MNEMA v2 — Adaptive State Core (ASC)

The most novel component of the MNEMA architecture.

What it is:
    A small, continuously-updated internal state vector that captures
    MNEMA's behavioral tendencies — who she is becoming, not just what
    she remembers. Think of it as her personality in motion.

What makes it different from everything else:
    - No backpropagation. No gradient descent. No weight changes.
    - Learning occurs via state evolution rules — like a recurrent
      cognitive state, not a neural network update.
    - Updates every single turn, not on a schedule.
    - Persists across sessions — MNEMA's behavioral character
      accumulates over weeks and months of use.

The state vector (64 dimensions, grouped into 8 behavioral axes):

    CURIOSITY       — how actively interested MNEMA is in this person
    WARMTH          — emotional closeness and care in responses
    FORMALITY       — casual vs formal register
    VERBOSITY       — tendency toward concise vs elaborate responses
    CONFIDENCE      — how certain MNEMA is in her memories and responses
    PLAYFULNESS     — humor, lightness, wit in responses
    DEPTH           — preference for surface vs philosophical/deep discussion
    CAUTION         — how careful MNEMA is about uncertain claims

Update rules (fired every turn, no backpropagation):
    1. RELEVANCE SIGNAL   — did the retrieved memories match well?
                           → boosts CONFIDENCE and CURIOSITY
    2. SURPRISE SIGNAL    — was the user's message unexpected?
                           → temporarily boosts DEPTH and CAUTION
    3. GOAL FEEDBACK      — which goals improved/declined this turn?
                           → adjusts axes linked to each goal
    4. TONE SIGNAL        — what was the user's emotional register?
                           → shifts WARMTH, FORMALITY, PLAYFULNESS
    5. MOMENTUM           — state drifts back toward baseline slowly
                           → prevents runaway personality shifts

Integration into inference:
    The ASC state is summarized as a behavioral guidance string:
    "Your current tendencies: warm and curious, moderately formal,
     prefer concise responses, slightly cautious about uncertain claims."
    This gets injected into the thinking prompt, shaping MNEMA's
    internal monologue and ultimately her spoken responses.
"""

import sqlite3
import time
import json
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Behavioral axes ───────────────────────────────────────────────────────────

AXES = [
    "curiosity",      # 0  — interest in the person and topics
    "warmth",         # 1  — emotional closeness
    "formality",      # 2  — casual (0) vs formal (1)
    "verbosity",      # 3  — concise (0) vs elaborate (1)
    "confidence",     # 4  — certainty in memories and responses
    "playfulness",    # 5  — humor and lightness
    "depth",          # 6  — surface (0) vs philosophical (1)
    "caution",        # 7  — careful about uncertain claims
]

N_AXES = len(AXES)

# Baseline state — where the state drifts back to slowly
BASELINE = np.array([
    0.6,   # curiosity     — naturally curious
    0.7,   # warmth        — warm but not excessive
    0.3,   # formality     — tends casual
    0.4,   # verbosity     — leans concise
    0.7,   # confidence    — fairly confident
    0.5,   # playfulness   — balanced
    0.5,   # depth         — balanced
    0.4,   # caution       — moderate caution
], dtype=np.float32)

# Momentum — how fast state drifts back to baseline (per turn)
# 0.02 = very slow drift, 0.1 = fast drift
MOMENTUM = 0.02

# Learning rate — how much signals shift the state per turn
LEARNING_RATE = 0.06

# State bounds
STATE_MIN = 0.05
STATE_MAX = 0.95


# ── Goal → axis linkage ───────────────────────────────────────────────────────
# Maps goal performance changes to which axes they influence

GOAL_AXIS_LINKS = {
    "minimize_corrections": {
        "confidence": +0.8,   # corrections hurt confidence
        "caution":    -0.6,   # corrections increase caution
    },
    "match_tone": {
        "warmth":      +0.7,
        "formality":   -0.3,  # matching tone usually means going more casual
        "playfulness": +0.4,
    },
    "be_concise": {
        "verbosity":  -0.8,   # conciseness goal suppresses verbosity
    },
    "remember_context": {
        "confidence": +0.6,   # successful memory use boosts confidence
        "curiosity":  +0.4,
    },
    "build_trust": {
        "warmth":     +0.5,
        "caution":    -0.3,
    },
}

# ── Tone signal → axis influence ──────────────────────────────────────────────

TONE_SIGNALS = {
    "casual": {
        "formality":   -0.3,
        "playfulness": +0.3,
        "verbosity":   -0.2,
    },
    "serious": {
        "formality":   +0.3,
        "depth":       +0.4,
        "playfulness": -0.2,
    },
    "curious": {
        "curiosity":   +0.4,
        "depth":       +0.3,
        "verbosity":   +0.2,
    },
    "frustrated": {
        "caution":     +0.4,
        "warmth":      +0.3,   # more warmth when frustrated
        "verbosity":   -0.3,   # be more concise
    },
    "positive": {
        "warmth":      +0.3,
        "playfulness": +0.2,
        "confidence":  +0.2,
    },
}


@dataclass
class ASCUpdateResult:
    """The result of a single ASC update — for logging and display."""
    turn: int
    previous_state: np.ndarray
    new_state: np.ndarray
    signals_applied: list[str]
    dominant_axes: list[str]      # top 3 axes with highest values
    behavioral_summary: str       # natural language description


class AdaptiveStateCore:
    """
    MNEMA's continuously evolving behavioral state.
    Updates every turn via signal-driven state evolution.
    No backpropagation required.
    """

    def __init__(self, db_path: str = "./data/memory_graph.db"):
        self.db_path = db_path
        self.state = BASELINE.copy()
        self._init_db()
        self._load_state()

    # ── Database ─────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS asc_state (
                id          INTEGER PRIMARY KEY DEFAULT 1,
                state_vector TEXT NOT NULL,
                turn_count  INTEGER DEFAULT 0,
                updated_at  REAL
            );

            CREATE TABLE IF NOT EXISTS asc_history (
                id          TEXT PRIMARY KEY,
                turn        INTEGER,
                state_vector TEXT,
                signals     TEXT,
                summary     TEXT,
                timestamp   REAL
            );
        """)
        conn.commit()
        conn.close()

    def _load_state(self):
        """Load persisted state from DB."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT state_vector FROM asc_state WHERE id = 1"
        ).fetchone()
        conn.close()

        if row:
            try:
                loaded = np.array(json.loads(row[0]), dtype=np.float32)
                if len(loaded) == N_AXES:
                    self.state = loaded
            except (json.JSONDecodeError, ValueError):
                self.state = BASELINE.copy()

    def _save_state(self, turn_count: int = 0):
        """Persist current state to DB."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO asc_state
            (id, state_vector, turn_count, updated_at)
            VALUES (1, ?, ?, ?)
        """, (json.dumps(self.state.tolist()), turn_count, time.time()))
        conn.commit()
        conn.close()

    # ── Core update ──────────────────────────────────────────────────────────

    def update(self, turn: int,
               user_message: str,
               goal_deltas: Optional[dict] = None,
               memory_match_score: float = 0.5,
               signals=None) -> ASCUpdateResult:
        """
        Update the state vector based on this turn's signals.
        This is the core of the ASC — called every single turn.

        Args:
            turn:               current turn number
            user_message:       raw user input
            goal_deltas:        {goal_id: delta} from goal layer
            memory_match_score: avg similarity of retrieved memories (0-1)
            signals:            TurnSignals from goal layer

        Returns ASCUpdateResult with the new state and what changed.
        """
        previous = self.state.copy()
        delta = np.zeros(N_AXES, dtype=np.float32)
        signals_applied = []

        # ── Signal 1: Memory relevance ────────────────────────────────────────
        # Good memory match → boost confidence and curiosity
        if memory_match_score > 0.7:
            delta[AXES.index("confidence")] += 0.15 * LEARNING_RATE
            delta[AXES.index("curiosity")]  += 0.10 * LEARNING_RATE
            signals_applied.append(f"memory_match={memory_match_score:.2f}↑")
        elif memory_match_score < 0.3:
            delta[AXES.index("confidence")] -= 0.2 * LEARNING_RATE
            delta[AXES.index("caution")]    += 0.3 * LEARNING_RATE
            signals_applied.append(f"memory_match={memory_match_score:.2f}↓")

        # ── Signal 2: Surprise (message length and content novelty) ──────────
        word_count = len(user_message.split())
        is_question = "?" in user_message
        is_long = word_count > 25

        if is_long and is_question:
            # Long questions = high depth/complexity
            delta[AXES.index("depth")]    += 0.25 * LEARNING_RATE
            delta[AXES.index("caution")]  += 0.15 * LEARNING_RATE
            signals_applied.append("surprise=complex_question")
        elif word_count < 5:
            # Very short messages = casual register
            delta[AXES.index("formality")]   -= 0.2 * LEARNING_RATE
            delta[AXES.index("verbosity")]   -= 0.15 * LEARNING_RATE
            signals_applied.append("surprise=very_short")

        # ── Signal 3: Goal feedback ───────────────────────────────────────────
        if goal_deltas:
            for goal_id, delta_val in goal_deltas.items():
                if goal_id in GOAL_AXIS_LINKS and delta_val != 0:
                    direction = 1 if delta_val > 0 else -1
                    magnitude = min(abs(delta_val), 0.2)
                    for axis, influence in GOAL_AXIS_LINKS[goal_id].items():
                        idx = AXES.index(axis)
                        delta[idx] += direction * magnitude * influence * LEARNING_RATE
                    if abs(delta_val) > 0.05:
                        signals_applied.append(
                            f"goal_{goal_id}={'↑' if delta_val > 0 else '↓'}"
                        )

        # ── Signal 4: Tone signals ────────────────────────────────────────────
        if signals:
            if signals.correction:
                # CHANGE — direct hard hit on confidence, not through goal pathway
                delta[AXES.index("confidence")] -= 0.12   # direct, no learning rate scaling
                delta[AXES.index("caution")]    += 0.10
                delta[AXES.index("warmth")]     += 0.04   # warmer when corrected
                delta[AXES.index("verbosity")]  -= 0.04   # more concise
                signals_applied.append("tone=correction")
            elif signals.positive:
                delta[AXES.index("confidence")] += 0.06
                delta[AXES.index("warmth")]     += 0.04
                delta[AXES.index("playfulness")] += 0.03
                signals_applied.append("tone=positive")

        # Detect question tone
        if is_question and word_count > 8:
            for axis, influence in TONE_SIGNALS["curious"].items():
                delta[AXES.index(axis)] += influence * LEARNING_RATE * 0.5
            signals_applied.append("tone=curious")

        # ── Signal 5: Momentum — drift back toward baseline ───────────────────
        baseline_pull = (BASELINE - self.state) * MOMENTUM
        delta += baseline_pull
        signals_applied.append("momentum")

        # ── Apply delta with bounds ───────────────────────────────────────────
        self.state = np.clip(self.state + delta, STATE_MIN, STATE_MAX)

        # ── Build result ──────────────────────────────────────────────────────
        dominant_axes = self._get_dominant_axes()
        summary = self._build_behavioral_summary()

        result = ASCUpdateResult(
            turn=turn,
            previous_state=previous,
            new_state=self.state.copy(),
            signals_applied=signals_applied,
            dominant_axes=dominant_axes,
            behavioral_summary=summary
        )

        # Persist every 5 turns to reduce DB writes
        if turn % 5 == 0:
            self._save_state(turn)
            self._log_history(turn, signals_applied, summary)

        return result

    # ── State reading ─────────────────────────────────────────────────────────

    def get_axis(self, axis: str) -> float:
        """Get current value for a named axis."""
        return float(self.state[AXES.index(axis)])

    def get_behavioral_guidance(self) -> str:
        """
        Returns natural language behavioral guidance for injection
        into the thinking prompt. This shapes MNEMA's internal monologue.
        """
        return self._build_behavioral_summary()

    def _get_dominant_axes(self) -> list[str]:
        """Return the 3 axes with highest current values."""
        indexed = sorted(enumerate(self.state), key=lambda x: x[1], reverse=True)
        return [AXES[i] for i, _ in indexed[:3]]

    def _build_behavioral_summary(self) -> str:
        """
        Build natural language description of current behavioral state.
        This is what gets injected into MNEMA's thinking prompt.
        """
        parts = []

        # Warmth
        w = self.get_axis("warmth")
        if w > 0.75:
            parts.append("genuinely warm")
        elif w < 0.35:
            parts.append("somewhat reserved")

        # Curiosity
        c = self.get_axis("curiosity")
        if c > 0.7:
            parts.append("actively curious about this person")
        elif c < 0.3:
            parts.append("somewhat detached")

        # Formality
        f = self.get_axis("formality")
        if f < 0.3:
            parts.append("casual and relaxed")
        elif f > 0.7:
            parts.append("tending formal")

        # Verbosity
        v = self.get_axis("verbosity")
        if v < 0.3:
            parts.append("preferring brevity")
        elif v > 0.7:
            parts.append("inclined to elaborate")

        # Confidence
        conf = self.get_axis("confidence")
        if conf > 0.8:
            parts.append("confident in memory")
        elif conf < 0.4:
            parts.append("uncertain — qualify claims")

        # Caution
        caut = self.get_axis("caution")
        if caut > 0.7:
            parts.append("cautious about uncertain things")

        # Playfulness
        p = self.get_axis("playfulness")
        if p > 0.7:
            parts.append("feeling playful")

        # Depth
        d = self.get_axis("depth")
        if d > 0.7:
            parts.append("drawn toward deeper discussion")
        elif d < 0.3:
            parts.append("keeping things surface-level")

        if not parts:
            return "balanced and attentive"

        return ", ".join(parts)

    def _log_history(self, turn: int, signals: list, summary: str):
        """Log state snapshot to history table."""
        import uuid
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO asc_history
            (id, turn, state_vector, signals, summary, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), turn,
              json.dumps(self.state.tolist()),
              json.dumps(signals), summary, time.time()))
        conn.commit()
        conn.close()

    # ── Display ───────────────────────────────────────────────────────────────

    def display_state(self) -> str:
        """Formatted state display for the 'asc' terminal command."""
        lines = ["\n  ── Adaptive State Core ──"]
        for i, axis in enumerate(AXES):
            val = float(self.state[i])
            base = float(BASELINE[i])
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            diff = val - base
            arrow = " ↑" if diff > 0.05 else (" ↓" if diff < -0.05 else "  ")
            lines.append(f"  {bar} {val:.2f}{arrow} {axis}")

        lines.append(f"\n  Behavioral summary:")
        lines.append(f"    {self._build_behavioral_summary()}")
        lines.append(f"\n  Dominant axes: {', '.join(self._get_dominant_axes())}")
        return "\n".join(lines)
