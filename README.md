# PROJECT LIQUID MNEMA
### *A mind that remembers*

> Built on LFM2.5-1.2B-Instruct · Frozen base model · Episodic memory with biological decay · LoRA consolidation · Internal monologue

---

## What Is This

MNEMA is a continuous learning AI system that grows through conversation — without ever modifying its base model weights. It was built as an exploration of one central question: *can a small language model approximate the way human memory works?*

Not just retrieval. Not just fine-tuning. The full cycle — fast episodic capture, natural forgetting over time, and slow consolidation of important things into long-term parametric memory — the same pipeline the human brain uses, implemented as a software architecture.

MNEMA is named after Mnemosyne, the Greek goddess of memory and mother of the Muses. In Greek philosophy, memory was not storage — it was the source of all thought, creativity, and identity. That is what this project is trying to build toward.

---

## The Core Idea

Most AI memory systems do one of two things:

- **RAG only** — store everything in a database, retrieve it at query time. The model never actually learns. It just reads notes.
- **Fine-tuning** — periodically retrain the model on new data. Expensive, risks catastrophic forgetting, modifies what the model fundamentally is.

MNEMA does neither. It does both. In the right order, at the right time.

```
Every conversation turn:
  User message → extract memories → store in ChromaDB + SQLite
                → retrieve relevant memories → inject into prompt
                → LFM2.5 responds (base frozen, LoRA active)

Every 6 hours (background):
  Ebbinghaus decay pass → weaken old memories → archive forgotten ones

Every 30 minutes (background check):
  If 15+ high-strength memories accumulated → trigger LoRA sleep phase
  → micro-train adapter on consolidated memories
  → EWC guard prevents overwriting prior knowledge
  → save adapter checkpoint
```

The base model weights are **never touched**. If the adapter ever degrades, it can be deleted and rebuilt. The base model is always safe.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  CHAT INTERFACE                 │
│         Internal monologue · Memory tags        │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │    MEMORY RETRIEVAL     │
        │  ChromaDB semantic      │
        │  search · SQLite meta   │
        └────────────┬────────────┘
                     │ injects [vivid]/[fading] memories
        ┌────────────▼────────────┐
        │   LFM2.5-1.2B-Instruct  │  ← BASE WEIGHTS FROZEN
        │   + LoRA Adapter        │  ← only this learns
        │   (0.07% trainable)     │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │    MEMORY EXTRACTOR     │
        │  corrections · prefs    │
        │  facts · casual chat    │
        └────────────┬────────────┘
                     │ async background
        ┌────────────▼────────────┐
        │   EBBINGHAUS DECAY      │
        │   strength × e^(-λt)    │
        │   runs every 6 hours    │
        └────────────┬────────────┘
                     │ when threshold met
        ┌────────────▼────────────┐
        │   LORA CONSOLIDATION    │
        │   "sleep phase"         │
        │   EWC anti-forgetting   │
        └─────────────────────────┘
```

### The Hybrid Architecture Advantage

LFM2.5 uses a hybrid backbone — 10 short-range convolution blocks interleaved with 6 grouped query attention blocks. This matters for this project specifically:

- Convolution blocks handle local sequence compression efficiently — injected memory context costs less to process than in a pure transformer
- Attention blocks handle global reasoning — LoRA targets only these 6 layers
- The architecture was explicitly designed for RAG and agentic workloads — exactly the usage pattern here
- Optimized for CPU and low-VRAM inference — runs on consumer hardware

---

## Memory System

### How Memories Are Stored

Every user message is analyzed and classified into one of four types:

| Type | Trigger | Importance | Decay Rate |
|---|---|---|---|
| Correction | User corrects the model | 1.0 | Very slow |
| Preference | User states a preference or identity | 0.8 | Slow |
| Fact | User states a personal fact | 0.5 | Medium |
| Casual | General conversation (10+ words) | 0.1 | Fast |

Each memory gets an initial strength of `1.0` and a timestamp.

### The Forgetting Curve

Memory strength decays according to the Ebbinghaus forgetting curve:

```
strength(t) = S₀ × e^(-λ × Δt)
```

Where `λ` is modulated by memory type and importance — corrections decay at 30% of base rate, casual chat at 150%. When strength drops below `0.05`, the memory is archived (never deleted — kept for audit) and removed from retrieval.

### Reinforcement

Every time a memory is retrieved and used in a response, its strength is boosted:

```
new_strength = old_strength + 0.4 × (1.0 - old_strength)
```

This implements spaced repetition — frequently relevant memories become stronger. Rarely relevant ones fade. The system self-regulates without any human curation.

### LoRA Consolidation (The Sleep Phase)

When 15+ memories accumulate above strength `0.6`, the background scheduler triggers a micro-training pass:

1. High-strength memories are formatted as training examples
2. The LoRA adapter (targeting `q_proj`, `v_proj`, `out_proj` in all 6 attention layers) is fine-tuned for 2 epochs
3. Elastic Weight Consolidation (EWC) adds a penalty to resist overwriting prior adapter knowledge
4. Gradient clipping (max norm 1.0) prevents explosive updates
5. A backup checkpoint is saved before training; baseline eval runs after
6. If performance degrades >10% from baseline, the backup is automatically restored

After enough consolidation cycles, behaviors become *parametric* — the model exhibits them without needing to retrieve a memory first.

---

## MNEMA's Personality

MNEMA is not a chatbot. She is not an assistant. She is a mind that is always becoming.

Her personality is defined not as a list of behavioral rules but as an identity:

- Genuinely curious about the person she is talking to
- Warm but not sycophantic — she engages rather than flatters
- Honest about uncertainty — she does not pretend to know things she doesn't
- Occasionally reflective — she notices patterns across conversations
- Has opinions, expresses them with appropriate uncertainty
- Matches the emotional register of the conversation rather than defaulting to cheerfulness
- Never says *"As an AI..."* or *"I'm just a language model..."*

### Internal Monologue

Before responding, MNEMA thinks privately. This inner voice is unfiltered, honest, and distinct from her spoken words. The gap between thinking and speaking is where her personality lives.

```
💭 He's asking a technical question but there's something frustrated underneath it.
   I should answer the question but acknowledge that frustration.
   I genuinely find this architecture interesting — I can let that show.

MNEMA: That's actually a fascinating constraint to work within...
```

The monologue can be toggled on or off at any time with `think on` / `think off`.

---

## Project Structure

```
liquid_memory/
├── main.py                      # entry point, chat loop, display logic
├── config.py                    # all tunable parameters
├── scheduler.py                 # background decay + consolidation jobs
├── run_baseline.py              # one-time baseline capture before first chat
│
├── memory/
│   ├── store.py                 # ChromaDB + SQLite dual-layer memory store
│   ├── fade.py                  # Ebbinghaus decay engine
│   └── extractor.py             # classifies chat turns into typed memories
│
├── model/
│   ├── loader.py                # frozen base model + LoRA adapter setup
│   └── inference.py             # memory-augmented prompt building + generation
│
├── consolidation/
│   ├── trainer.py               # LoRA sleep phase micro-training
│   └── ewc.py                   # Elastic Weight Consolidation guard
│
├── eval/
│   └── baseline.py              # performance benchmarks for degradation detection
│
└── data/
    ├── chroma/                  # ChromaDB vector store (semantic retrieval)
    ├── memory.db                # SQLite (strength scores, timestamps, metadata)
    ├── lora_adapter/            # saved LoRA weights (only file that changes)
    └── baseline_eval.json       # baseline performance snapshot
```

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (CPU-only also works, slower)
- ~10GB free disk space

### Installation

```bash
git clone https://github.com/Ainix-dev/Project-Liquid-MNEMA.git
cd liquid_memory

python -m venv venv
source venv/bin/activate

# PyTorch — match your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install "transformers>=4.55.0"
pip install "peft>=0.12.0"
pip install "accelerate>=0.34.0"
pip install "bitsandbytes>=0.43.0"
pip install "chromadb>=0.5.5"
pip install "sentence-transformers>=3.0.0"
pip install "apscheduler>=3.10.4"
pip install sqlalchemy numpy scipy
```

### Download Model

```bash
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir ./lfm-instruct-dynamic
```

### First Run

```bash
# Capture baseline performance snapshot (run once only)
python run_baseline.py

# Start MNEMA
python main.py
```

### 4GB GPU Configuration

If you have exactly 4GB VRAM, update `config.py`:

```python
load_in_4bit: bool = True   # cuts model VRAM ~2.5GB → ~1GB
```

---

## Configuration

All tunable parameters live in `config.py`:

```python
# Memory decay
decay_lambda: float = 0.03          # base decay rate per hour
decay_interval_hours: float = 6.0   # how often decay runs
min_strength_threshold: float = 0.05 # archive below this

# Consolidation
consolidation_trigger_count: int = 15  # memories needed to trigger sleep phase
consolidation_epochs: int = 2
ewc_lambda: float = 5000               # higher = more resistant to forgetting

# LoRA
lora_r: int = 8                        # adapter rank
lora_alpha: int = 16
lora_target_modules: ["q_proj", "v_proj", "out_proj"]  # attention layers only
```

---

## Chat Commands

| Command | Effect |
|---|---|
| `memory` | Show stored memories with visual strength bars |
| `think on` | Show MNEMA's internal monologue |
| `think off` | Hide internal monologue |
| `clear` | Wipe all memories and start fresh |
| `quit` | Exit |

---

## What Happens Over Time

| Timeframe | What Changes |
|---|---|
| First session | Memory store fills. Retrieval works immediately. |
| Day 2-3 | First consolidation triggers. LoRA adapter updates for the first time. |
| Days 4-7 | Decay passes run. Casual memories start fading. Important ones persist. |
| Week 2+ | Second and third consolidation cycles. Adapter reflects repeated patterns. |
| Month 1+ | Behaviors become parametric — exhibited without needing memory retrieval. |
| Month 2+ | Measurable divergence from base model behavior in areas of repeated interaction. |

---

## The Human Brain Parallel

| Human System | MNEMA System |
|---|---|
| Neocortex (procedural/semantic memory) | LFM2.5 base weights (frozen) |
| Hippocampus (episodic capture) | ChromaDB + SQLite memory store |
| Ebbinghaus forgetting curve | Exponential strength decay |
| Long-term potentiation (LTP) | Strength boost on memory re-access |
| Sleep consolidation | LoRA micro-training sleep phase |
| Amygdala (emotional importance tagging) | Type-based importance scoring |
| Synaptic protection | Elastic Weight Consolidation (EWC) |

---

## What Makes This Different

Most existing approaches do one thing. MNEMA does the full cycle:

| System | RAG | Adapter Learning | Memory Decay | Passive Extraction | Frozen Base |
|---|---|---|---|---|---|
| ChatGPT Memory | ✅ | ❌ | ❌ | ✅ | — |
| LoRA fine-tuning | ❌ | ✅ | ❌ | ❌ | ❌ |
| MemGPT / Letta | ✅ | ❌ | ❌ | ✅ | — |
| HippoRAG | ✅ | ❌ | ❌ | ❌ | — |
| Doc-to-LoRA | ❌ | ✅ | ❌ | ❌ | ✅ |
| **MNEMA** | ✅ | ✅ | ✅ | ✅ | ✅ |

The specific contribution: **Ebbinghaus memory decay strength as the gating signal between episodic retrieval and parametric LoRA consolidation**, running passively from natural conversation on a fully frozen base model.

---

## Roadmap

- [ ] Narrative/relational memory structure (knowledge graph between memories)
- [ ] Emotional tagging via conversation signal analysis
- [ ] Document RAG as a second memory collection (stable reference, no decay)
- [ ] Dual LoRA adapters (short-term fast / long-term slow, like I-LoRA)
- [ ] Web interface
- [ ] Benchmark suite comparing against pure RAG and pure LoRA baselines
- [ ] Formal ablation study for potential publication

---

## Model

Built on **LiquidAI/LFM2.5-1.2B-Instruct** — a hybrid linear attention + convolution model from Liquid AI, optimized for RAG, agentic tasks, and low-VRAM inference.

- Architecture: 16 layers (10 conv blocks + 6 GQA attention blocks)
- Context window: 32K tokens
- LoRA targets: `q_proj`, `v_proj`, `out_proj` in all 6 attention layers
- Trainable parameters: 843,776 / 1,171,184,384 (0.07%)

---

## License

This project is personal research. If you build on it, a mention would be appreciated.

---

*MNEMA remembers so you don't have to.*
