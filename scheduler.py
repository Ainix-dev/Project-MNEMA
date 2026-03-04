# scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from memory.fade import EbbinghausDecay
from consolidation.trainer import consolidate
from config import cfg


class MemoryScheduler:
    def __init__(self, memory_store, model, tokenizer):
        self.store = memory_store
        self.model = model
        self.tokenizer = tokenizer
        self.decay_engine = EbbinghausDecay(memory_store)
        self.ewc_state = None
        self.scheduler = BackgroundScheduler()

    def start(self):
        # Decay pass every N hours
        self.scheduler.add_job(
            self._run_decay,
            "interval",
            hours=cfg.decay_interval_hours,
            id="memory_decay"
        )

        # Consolidation check every 30 minutes
        self.scheduler.add_job(
            self._check_consolidation,
            "interval",
            minutes=30,
            id="consolidation_check"
        )

        self.scheduler.start()
        print(f"[Scheduler] Started. Decay every {cfg.decay_interval_hours}h, consolidation check every 30min.")

    def pause(self):
        """Pause background tasks during high memory pressure."""
        if self.scheduler.running:
            self.scheduler.pause()

    def resume(self):
        """Resume background tasks when memory pressure eases."""
        if self.scheduler.running:
            self.scheduler.resume()

    def stop(self):
        self.scheduler.shutdown()

    def _run_decay(self):
        print("[Scheduler] Running memory decay pass...")
        self.decay_engine.run_decay_pass()

    def _check_consolidation(self):
        candidates = self.store.get_consolidation_candidates()
        if len(candidates) >= cfg.consolidation_trigger_count:
            print(f"[Scheduler] Consolidation threshold met ({len(candidates)} candidates). Running sleep phase...")
            self.ewc_state = consolidate(
                self.model, self.tokenizer, self.store, self.ewc_state
            )
