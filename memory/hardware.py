# memory/hardware.py
"""
MNEMA v2 — Hardware-Aware Adaptation

Dynamically adjusts MNEMA's operations based on available
VRAM, RAM, and CPU load. Calibrated for GTX 1050 Ti (4GB VRAM).

Operation Tiers:
┌───────────┬──────────┬───────────────────────────────────────────────┐
│ Tier      │ VRAM     │ Behavior                                      │
├───────────┼──────────┼───────────────────────────────────────────────┤
│ FULL      │ > 1.2GB  │ Two-pass generation, full monologue, all ASC  │
│ REDUCED   │ 0.7-1.2GB│ Two-pass but shorter tokens, ASC still runs   │
│ MINIMAL   │ 0.4-0.7GB│ Single pass, no monologue, ASC still runs     │
│ EMERGENCY │ < 0.4GB  │ Single pass, shortest tokens, pause scheduler │
└───────────┴──────────┴───────────────────────────────────────────────┘

What gets adjusted per tier:
    max_new_tokens          — tokens for thinking pass
    response_tokens         — tokens for response pass
    show_thinking_allowed   — whether monologue is allowed
    top_k_memories          — how many memories to retrieve
    history_budget          — how many tokens of history to keep
    scheduler_pause         — whether to pause background tasks

Monitors every N turns and on demand.
Logs pressure events to DB for trend analysis.
"""

import time
import sqlite3
import json
import torch
import psutil
from dataclasses import dataclass


# ── Hardware profile — GTX 1050 Ti ───────────────────────────────────────────

HARDWARE_PROFILE = {
    "gpu_name":       "NVIDIA GeForce GTX 1050 Ti",
    "vram_total_gb":  4.23,
    "ram_total_gb":   16.69,
    "cpu_cores":      6,

    # VRAM thresholds for tier switching
    "vram_full_gb":     1.2,   # above this = FULL
    "vram_reduced_gb":  0.7,   # above this = REDUCED
    "vram_minimal_gb":  0.4,   # above this = MINIMAL
                               # below minimal = EMERGENCY

    # RAM thresholds
    "ram_pressure_gb":  3.0,   # below this = RAM pressure warning
    "ram_critical_gb":  1.5,   # below this = RAM critical

    # CPU thresholds
    "cpu_pressure_pct": 80,    # above this = CPU pressure
}


# ── Operation configs per tier ────────────────────────────────────────────────

TIER_CONFIGS = {
    "FULL": {
        "thinking_tokens":      250,
        "response_tokens":      400,
        "show_thinking_allowed": True,
        "top_k_memories":       5,
        "history_budget":       1800,
        "scheduler_pause":      False,
        "description": "All systems active — comfortable headroom",
    },
    "REDUCED": {
        "thinking_tokens":      150,
        "response_tokens":      300,
        "show_thinking_allowed": True,
        "top_k_memories":       4,
        "history_budget":       1400,
        "scheduler_pause":      False,
        "description": "Slightly constrained — shorter generation",
    },
    "MINIMAL": {
        "thinking_tokens":      0,     # no monologue pass
        "response_tokens":      250,
        "show_thinking_allowed": False,
        "top_k_memories":       3,
        "history_budget":       1000,
        "scheduler_pause":      False,
        "description": "Low VRAM — monologue disabled, single pass only",
    },
    "EMERGENCY": {
        "thinking_tokens":      0,
        "response_tokens":      180,
        "show_thinking_allowed": False,
        "top_k_memories":       2,
        "history_budget":       600,
        "scheduler_pause":      True,
        "description": "Critical VRAM — minimal mode, scheduler paused",
    },
}


@dataclass
class HardwareSnapshot:
    """Point-in-time hardware reading."""
    timestamp: float
    vram_free_gb: float
    vram_used_gb: float
    vram_pct_used: float
    ram_free_gb: float
    ram_pct_used: float
    cpu_pct: float
    tier: str
    config: dict


class HardwareMonitor:
    """
    Monitors hardware state and provides adaptive operation configs.
    Calibrated for GTX 1050 Ti with 4GB VRAM.
    """

    def __init__(self, db_path: str = "./data/memory_graph.db",
                 check_every_n_turns: int = 3):
        self.db_path = db_path
        self.check_every_n_turns = check_every_n_turns
        self.current_tier = "FULL"
        self.current_config = TIER_CONFIGS["FULL"].copy()
        self._last_snapshot = None
        self._turn_counter = 0
        self._init_db()
        # Take initial reading
        self.update()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS hardware_log (
                id          TEXT PRIMARY KEY,
                timestamp   REAL,
                tier        TEXT,
                vram_free   REAL,
                ram_free    REAL,
                cpu_pct     REAL,
                note        TEXT
            );
        """)
        conn.commit()
        conn.close()

    # ── Core monitoring ───────────────────────────────────────────────────────

    def update(self, force: bool = False) -> HardwareSnapshot:
        """
        Take a hardware snapshot and update tier if needed.
        Only re-reads hardware every check_every_n_turns turns
        unless forced.
        """
        self._turn_counter += 1
        if not force and self._turn_counter % self.check_every_n_turns != 0:
            return self._last_snapshot

        snapshot = self._read_hardware()
        old_tier = self.current_tier

        # Determine new tier
        new_tier = self._determine_tier(snapshot)

        if new_tier != old_tier:
            self.current_tier = new_tier
            self.current_config = TIER_CONFIGS[new_tier].copy()
            self._log_tier_change(snapshot, old_tier, new_tier)
            print(f"\n  [Hardware] Tier changed: {old_tier} → {new_tier} "
                  f"({TIER_CONFIGS[new_tier]['description']})\n")

        self._last_snapshot = snapshot
        return snapshot

    def _read_hardware(self) -> HardwareSnapshot:
        """Read current hardware state."""
        # VRAM
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            vram_free_gb  = free_bytes / 1e9
            vram_used_gb  = (total_bytes - free_bytes) / 1e9
            vram_pct_used = (vram_used_gb / (total_bytes / 1e9)) * 100
        else:
            vram_free_gb  = 0.0
            vram_used_gb  = 0.0
            vram_pct_used = 100.0

        # RAM
        ram = psutil.virtual_memory()
        ram_free_gb  = ram.available / 1e9
        ram_pct_used = ram.percent

        # CPU (non-blocking — 0.1s interval)
        cpu_pct = psutil.cpu_percent(interval=0.1)

        tier = self._determine_tier_from_values(vram_free_gb, ram_free_gb)

        return HardwareSnapshot(
            timestamp=time.time(),
            vram_free_gb=vram_free_gb,
            vram_used_gb=vram_used_gb,
            vram_pct_used=vram_pct_used,
            ram_free_gb=ram_free_gb,
            ram_pct_used=ram_pct_used,
            cpu_pct=cpu_pct,
            tier=tier,
            config=TIER_CONFIGS[tier].copy()
        )

    def _determine_tier(self, snapshot: HardwareSnapshot) -> str:
        return snapshot.tier

    def _determine_tier_from_values(self, vram_free_gb: float,
                                     ram_free_gb: float) -> str:
        p = HARDWARE_PROFILE

        # RAM critical overrides everything
        if ram_free_gb < p["ram_critical_gb"]:
            return "EMERGENCY"

        # VRAM-based tiering
        if vram_free_gb >= p["vram_full_gb"]:
            return "FULL"
        elif vram_free_gb >= p["vram_reduced_gb"]:
            return "REDUCED"
        elif vram_free_gb >= p["vram_minimal_gb"]:
            return "MINIMAL"
        else:
            return "EMERGENCY"

    def _log_tier_change(self, snapshot: HardwareSnapshot,
                          old_tier: str, new_tier: str):
        import uuid
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO hardware_log
            (id, timestamp, tier, vram_free, ram_free, cpu_pct, note)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (str(uuid.uuid4()), snapshot.timestamp, new_tier,
              snapshot.vram_free_gb, snapshot.ram_free_gb,
              snapshot.cpu_pct,
              f"{old_tier}→{new_tier}"))
        conn.commit()
        conn.close()

    # ── Config access ─────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return current operation config for this tier."""
        return self.current_config

    def thinking_allowed(self, user_requested: bool) -> bool:
        """
        Whether monologue pass is allowed given hardware state.
        Respects user preference but overrides if VRAM is critical.
        """
        return user_requested and self.current_config["show_thinking_allowed"]

    def get_tokens(self) -> tuple[int, int]:
        """Return (thinking_tokens, response_tokens) for current tier."""
        return (
            self.current_config["thinking_tokens"],
            self.current_config["response_tokens"]
        )

    def get_history_budget(self) -> int:
        return self.current_config["history_budget"]

    def get_top_k(self) -> int:
        return self.current_config["top_k_memories"]

    def should_pause_scheduler(self) -> bool:
        return self.current_config["scheduler_pause"]

    # ── VRAM cache management ─────────────────────────────────────────────────

    def clear_vram_cache(self):
        """
        Clear CUDA cache — call between generation passes when under pressure.
        Safe to call anytime on 1050 Ti.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def vram_pressure(self) -> bool:
        """Quick check — are we below comfortable headroom?"""
        if not self._last_snapshot:
            return False
        return self._last_snapshot.vram_free_gb < HARDWARE_PROFILE["vram_full_gb"]

    # ── Display ───────────────────────────────────────────────────────────────

    def display_status(self) -> str:
        snap = self._read_hardware()
        tier_config = TIER_CONFIGS[snap.tier]

        # VRAM bar
        vram_used_pct = snap.vram_pct_used / 100
        vram_bar_len = int(vram_used_pct * 20)
        vram_bar = "█" * vram_bar_len + "░" * (20 - vram_bar_len)

        # RAM bar
        ram_used_pct = snap.ram_pct_used / 100
        ram_bar_len = int(ram_used_pct * 20)
        ram_bar = "█" * ram_bar_len + "░" * (20 - ram_bar_len)

        # CPU bar
        cpu_pct = snap.cpu_pct / 100
        cpu_bar_len = int(cpu_pct * 20)
        cpu_bar = "█" * cpu_bar_len + "░" * (20 - cpu_bar_len)

        tier_colors = {
            "FULL": "✅", "REDUCED": "🟡",
            "MINIMAL": "🟠", "EMERGENCY": "🔴"
        }
        icon = tier_colors.get(snap.tier, "●")

        lines = [
            f"\n  ── Hardware Status ──",
            f"  {icon} Tier: {snap.tier} — {tier_config['description']}",
            f"",
            f"  VRAM  {vram_bar} {snap.vram_used_gb:.2f}/{HARDWARE_PROFILE['vram_total_gb']}GB "
            f"({snap.vram_pct_used:.0f}%) — {snap.vram_free_gb:.2f}GB free",
            f"  RAM   {ram_bar} {snap.ram_pct_used:.0f}% used "
            f"— {snap.ram_free_gb:.1f}GB free",
            f"  CPU   {cpu_bar} {snap.cpu_pct:.0f}%",
            f"",
            f"  Active config:",
            f"    thinking tokens:  {tier_config['thinking_tokens']}",
            f"    response tokens:  {tier_config['response_tokens']}",
            f"    memory top-k:     {tier_config['top_k_memories']}",
            f"    history budget:   {tier_config['history_budget']} tokens",
            f"    monologue:        {'enabled' if tier_config['show_thinking_allowed'] else 'disabled (low VRAM)'}",
            f"    scheduler:        {'paused' if tier_config['scheduler_pause'] else 'running'}",
        ]
        return "\n".join(lines)
