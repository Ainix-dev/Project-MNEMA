# memory/metacog.py
"""
MNEMA v2 — Self-Modeling / Meta-Cognition

Tracks MNEMA's own performance, confidence, and error patterns.
This is what allows MNEMA to know what she doesn't know.

What it tracks:
    - Correction events (when, what topic, how often)
    - Confidence per memory type (how reliable is each kind of memory)
    - Error patterns (repeated mistakes on same topics)
    - Reliability score over time (overall self-assessment)
    - Response quality signals (positive/negative feedback)

What it does with that tracking:
    - Adjusts memory injection priority (less confident = more memory needed)
    - Flags topics where MNEMA has been repeatedly wrong
    - Provides self-awareness context to the internal monologue
    - Informs the Goal layer when to adjust weights

The meta-cognition state is injected into MNEMA's thinking prompt
so she can reason about her own reliability:
    "I've been corrected about Ken's job twice — I should be careful here"
    "I'm confident about his preferences — I've never been corrected on those"
"""

import sqlite3
import time
import json
from dataclasses import dataclass, field
from typing import Optional


# ── Confidence levels ─────────────────────────────────────────────────────────

CONFIDENCE_HIGH   = "high"      # never corrected on this topic/type
CONFIDENCE_MEDIUM = "medium"    # corrected once or twice
CONFIDENCE_LOW    = "low"       # corrected 3+ times
CONFIDENCE_UNSURE = "unsure"    # no data yet

# Correction count thresholds
CORRECTION_MEDIUM_THRESHOLD = 1
CORRECTION_LOW_THRESHOLD    = 3


@dataclass
class CorrectionEvent:
    """A single correction event."""
    turn: int
    user_message: str
    topic_hint: str       # first 60 chars of the corrected content
    memory_type: str
    timestamp: float


@dataclass
class MetaCogState:
    """
    MNEMA's current self-assessment.
    Injected into thinking prompt when show_thinking is on.
    """
    reliability_score: float          # 0.0 - 1.0
    correction_count: int
    recent_corrections: list[str]     # last 3 correction topics
    confident_types: list[str]        # memory types with no corrections
    weak_types: list[str]             # memory types with many corrections
    self_note: str                    # natural language self-assessment


class MetaCognition:
    """
    MNEMA's self-modeling engine.
    Tracks performance and provides self-awareness context.
    """

    def __init__(self, db_path: str = "./data/memory_graph.db"):
        self.db_path = db_path
        self._init_db()

    # ── Database ─────────────────────────────────────────────────────────────

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS correction_events (
                id           TEXT PRIMARY KEY,
                turn         INTEGER,
                user_message TEXT,
                topic_hint   TEXT,
                memory_type  TEXT DEFAULT 'unknown',
                timestamp    REAL
            );

            CREATE TABLE IF NOT EXISTS confidence_scores (
                memory_type     TEXT PRIMARY KEY,
                correction_count INTEGER DEFAULT 0,
                access_count     INTEGER DEFAULT 0,
                last_correction  REAL,
                confidence_level TEXT DEFAULT 'unsure',
                updated_at       REAL
            );

            CREATE TABLE IF NOT EXISTS reliability_log (
                id          TEXT PRIMARY KEY,
                score       REAL,
                note        TEXT,
                timestamp   REAL
            );
        """)
        conn.commit()
        conn.close()

    # ── Recording events ─────────────────────────────────────────────────────

    def record_correction(self, turn: int, user_message: str,
                          memory_type: str = "unknown") -> None:
        """
        Record that MNEMA was corrected.
        Updates confidence scores for the relevant memory type.
        """
        import uuid
        now = time.time()
        topic_hint = user_message[:80]

        conn = sqlite3.connect(self.db_path)

        # Log the correction event
        conn.execute("""
            INSERT INTO correction_events
            (id, turn, user_message, topic_hint, memory_type, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), turn, user_message,
              topic_hint, memory_type, now))

        # Update confidence for this memory type
        conn.execute("""
            INSERT INTO confidence_scores
            (memory_type, correction_count, access_count,
             last_correction, confidence_level, updated_at)
            VALUES (?, 1, 0, ?, 'medium', ?)
            ON CONFLICT(memory_type) DO UPDATE SET
                correction_count = correction_count + 1,
                last_correction = ?,
                confidence_level = CASE
                    WHEN correction_count + 1 >= ? THEN 'low'
                    WHEN correction_count + 1 >= ? THEN 'medium'
                    ELSE 'high'
                END,
                updated_at = ?
        """, (memory_type, now, now, now,
              CORRECTION_LOW_THRESHOLD,
              CORRECTION_MEDIUM_THRESHOLD, now))

        conn.commit()
        conn.close()

        # Update reliability score
        self._update_reliability()

    def record_access(self, memory_type: str) -> None:
        """Record that a memory of this type was accessed successfully."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO confidence_scores
            (memory_type, correction_count, access_count,
             last_correction, confidence_level, updated_at)
            VALUES (?, 0, 1, NULL, 'high', ?)
            ON CONFLICT(memory_type) DO UPDATE SET
                access_count = access_count + 1,
                updated_at = ?
        """, (memory_type, time.time(), time.time()))
        conn.commit()
        conn.close()

    def record_positive(self, turn: int) -> None:
        """Record a positive signal — user confirmed MNEMA was right."""
        self._update_reliability(boost=+0.02)

    # ── Reliability ───────────────────────────────────────────────────────────

    def _update_reliability(self, boost: float = -0.05) -> float:
        """Recalculate overall reliability score."""
        import uuid
        conn = sqlite3.connect(self.db_path)

        total_corrections = conn.execute(
            "SELECT COUNT(*) FROM correction_events"
        ).fetchone()[0]

        total_accesses = conn.execute(
            "SELECT SUM(access_count) FROM confidence_scores"
        ).fetchone()[0] or 1

        # Base reliability: ratio of correct to total interactions
        base = 1.0 - min(1.0, total_corrections / max(total_accesses, 1))

        # Recent corrections hurt more
        recent_corrections = conn.execute("""
            SELECT COUNT(*) FROM correction_events
            WHERE timestamp > ?
        """, (time.time() - 3600 * 24,)).fetchone()[0]  # last 24h

        recency_penalty = min(0.3, recent_corrections * 0.08)
        reliability = max(0.1, base - recency_penalty + boost)
        reliability = min(1.0, reliability)

        conn.execute("""
            INSERT INTO reliability_log (id, score, note, timestamp)
            VALUES (?, ?, ?, ?)
        """, (str(uuid.uuid4()), reliability,
              f"corrections={total_corrections}", time.time()))

        conn.commit()
        conn.close()
        return reliability

    def get_reliability_score(self) -> float:
        """Get current reliability score."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT score FROM reliability_log
            ORDER BY timestamp DESC LIMIT 1
        """).fetchone()
        conn.close()
        return row[0] if row else 0.85  # optimistic default

    # ── Confidence per type ───────────────────────────────────────────────────

    def get_confidence(self, memory_type: str) -> str:
        """Return confidence level for a memory type."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("""
            SELECT confidence_level FROM confidence_scores
            WHERE memory_type = ?
        """, (memory_type,)).fetchone()
        conn.close()
        return row[0] if row else CONFIDENCE_UNSURE

    def get_all_confidence(self) -> dict:
        """Return confidence levels for all tracked memory types."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT memory_type, confidence_level, correction_count, access_count
            FROM confidence_scores
        """).fetchall()
        conn.close()
        return {
            r[0]: {
                "confidence": r[1],
                "corrections": r[2],
                "accesses": r[3]
            } for r in rows
        }

    # ── Self-awareness state ──────────────────────────────────────────────────

    def get_state(self) -> MetaCogState:
        """
        Build MNEMA's current self-assessment.
        This gets injected into her thinking prompt.
        """
        conn = sqlite3.connect(self.db_path)

        reliability = self.get_reliability_score()

        correction_count = conn.execute(
            "SELECT COUNT(*) FROM correction_events"
        ).fetchone()[0]

        recent = conn.execute("""
            SELECT topic_hint FROM correction_events
            ORDER BY timestamp DESC LIMIT 3
        """).fetchall()
        recent_corrections = [r[0] for r in recent]

        conf_rows = conn.execute("""
            SELECT memory_type, confidence_level, correction_count
            FROM confidence_scores
        """).fetchall()
        conn.close()

        confident_types = [r[0] for r in conf_rows if r[1] == "high"]
        weak_types = [r[0] for r in conf_rows if r[1] == "low"]

        # Build natural language self-note
        self_note = self._build_self_note(
            reliability, correction_count,
            recent_corrections, weak_types
        )

        return MetaCogState(
            reliability_score=reliability,
            correction_count=correction_count,
            recent_corrections=recent_corrections,
            confident_types=confident_types,
            weak_types=weak_types,
            self_note=self_note
        )

    def _build_self_note(self, reliability: float, correction_count: int,
                          recent_corrections: list, weak_types: list) -> str:
        """
        Build a natural language self-assessment for the thinking prompt.
        This is what MNEMA reads about herself before responding.
        """
        notes = []

        if correction_count == 0:
            notes.append("I haven't been corrected yet — "
                         "I should stay confident but not overconfident.")
        elif correction_count <= 2:
            notes.append(f"I've been corrected {correction_count} time(s) — "
                         f"I'm still fairly reliable but should stay careful.")
        else:
            notes.append(f"I've been corrected {correction_count} times — "
                         f"I need to be more careful and qualify uncertain statements.")

        if recent_corrections:
            topics = "; ".join(recent_corrections[:2])
            notes.append(f"Recent corrections were about: {topics[:100]}.")

        if weak_types:
            notes.append(f"I'm least reliable on: {', '.join(weak_types)}.")

        if reliability < 0.5:
            notes.append("My overall reliability is low right now. "
                         "I should be more humble and check my assumptions.")
        elif reliability > 0.85:
            notes.append("My reliability has been good recently.")

        return " ".join(notes)

    # ── Error pattern detection ───────────────────────────────────────────────

    def get_repeated_error_topics(self, min_count: int = 2) -> list[str]:
        """
        Find topics where MNEMA has been corrected multiple times.
        These are areas where she should be especially careful.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT topic_hint, COUNT(*) as cnt
            FROM correction_events
            GROUP BY topic_hint
            HAVING cnt >= ?
            ORDER BY cnt DESC
        """, (min_count,)).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # ── Display ───────────────────────────────────────────────────────────────

    def display_summary(self) -> str:
        """Return a formatted summary for the 'metacog' terminal command."""
        state = self.get_state()
        conf = self.get_all_confidence()
        repeated = self.get_repeated_error_topics()

        lines = [
            f"  ── Meta-Cognition State ──",
            f"  Reliability score:  {state.reliability_score:.2f}",
            f"  Total corrections:  {state.correction_count}",
            "",
        ]

        if conf:
            lines.append("  Confidence by memory type:")
            for mem_type, data in conf.items():
                bar = {"high": "●●●", "medium": "●●○",
                       "low": "●○○", "unsure": "○○○"}.get(
                    data["confidence"], "○○○")
                lines.append(f"    {bar} {mem_type:<12} "
                             f"corrections={data['corrections']} "
                             f"accesses={data['accesses']}")
            lines.append("")

        if repeated:
            lines.append("  Repeated error topics:")
            for topic in repeated[:3]:
                lines.append(f"    ⚠ {topic[:70]}")
            lines.append("")

        if state.recent_corrections:
            lines.append("  Recent corrections:")
            for c in state.recent_corrections:
                lines.append(f"    • {c[:70]}")
            lines.append("")

        lines.append(f"  Self-assessment:")
        lines.append(f"    {state.self_note}")

        return "\n".join(lines)
