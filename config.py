from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    model_id: str = os.getenv("MODEL_PATH", "./lfm-instruct-dynamic")
    adapter_path: str = os.getenv("ADAPTER_PATH", "./data/lora_adapter")
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    load_in_4bit: bool = True

    # MEMORY DECAY
    decay_lambda: float = 0.03
    decay_interval_hours: float = 6.0
    min_strength_threshold: float = 0.05
    reinforcement_boost: float = 0.4
    max_strength: float = 1.0

    # IMPORTANCE SCORING
    importance_weights: dict = field(default_factory=lambda: {
        "correction": 1.0,
        "preference": 0.8,
        "fact": 0.5,
        "casual": 0.1,
    })

    # RETRIEVAL
    top_k_memories: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"

    # CONSOLIDATION
    consolidation_trigger_count: int = 15
    consolidation_min_strength: float = 0.6
    consolidation_epochs: int = 2
    consolidation_lr: float = 5e-5
    ewc_lambda: float = 5000

    # LoRA TARGETS — confirmed from layer_inspect.py output
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj",    # query projection (attention layers 2,5,8,10,12,14)
        "v_proj",    # value projection
        "out_proj",  # attention output (LFM2.5 uses out_proj NOT o_proj)
    ])

cfg = Config()
