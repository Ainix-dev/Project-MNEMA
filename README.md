<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=MNEMA&fontSize=80&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=A%20mind%20that%20remembers&descAlignY=60&descSize=20&descColor=aaa" width="100%"/>

<br/>

[![License](https://img.shields.io/badge/License-Apache_2.0-4A90D9?style=for-the-badge&logo=apache&logoColor=white)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Model](https://img.shields.io/badge/LFM2.5--1.2B-Instruct-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
[![Status](https://img.shields.io/badge/Status-Active_Research-22C55E?style=for-the-badge&logo=statuspage&logoColor=white)]()

<br/>

[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.55%2B-yellow?style=flat-square)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange?style=flat-square)](https://github.com/huggingface/peft)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-blueviolet?style=flat-square)](https://www.trychroma.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=github)](./CONTRIBUTING.md)

<br/><br/>

> *Born from liquid neural architecture, biological memory theory, and the forgetting curve.*
> *MNEMA is not a chatbot. She is a mind that is always becoming.*

<br/>

```
She remembers what matters.
She forgets what doesn't.
She grows with every conversation.
```

</div>

---

## ◈ What Is This

MNEMA is a **continuous learning AI system** that grows through conversation — without ever modifying its base model weights. It was built around one central question:

> *Can a small language model approximate the way human memory actually works?*

Not just retrieval. Not just fine-tuning. The **full biological cycle** — fast episodic capture, natural forgetting over time, reinforcement through re-access, and slow consolidation of important things into long-term parametric memory. The same pipeline the human brain uses, implemented as a software architecture on a frozen 1.2B model.

<br/>

<div align="center">

| | Pure RAG | Pure LoRA | MemGPT | **MNEMA** |
|---|:---:|:---:|:---:|:---:|
| Episodic memory retrieval | ✅ | ❌ | ✅ | ✅ |
| Parametric adapter learning | ❌ | ✅ | ❌ | ✅ |
| Biological memory decay | ❌ | ❌ | ❌ | ✅ |
| Passive extraction from chat | ❌ | ❌ | ✅ | ✅ |
| Base model fully frozen | — | ❌ | — | ✅ |
| Internal monologue | ❌ | ❌ | ❌ | ✅ |

</div>

---

## ◈ The Core Idea

Most AI memory systems do one of two things:

- **RAG only** — store everything, retrieve at query time. The model never actually learns. It just reads notes.
- **Fine-tuning** — periodically retrain. Expensive, risks catastrophic forgetting, modifies what the model fundamentally is.

MNEMA does **neither**. It does **both** — in the right order, at the right time.

```
Every turn:      extract memory → store → retrieve relevant → inject → respond

Every 6 hours:   Ebbinghaus decay pass → weaken old memories → archive forgotten

Every 30 min:    check consolidation threshold →
                 if 15+ strong memories → trigger LoRA sleep phase →
                 micro-train adapter → EWC guard → save checkpoint
```

The base model weights are **never touched.** If the adapter ever degrades, delete it and rebuild. The base is always safe.

---

## ◈ Architecture

<br/>

```
╔═════════════════════════════════════════════════════════╗
║                    CHAT INTERFACE                       ║
║         💭 Internal monologue  ·  📝 Memory tags        ║
╚══════════════════════╦══════════════════════════════════╝
                       ║
          ╔════════════▼════════════╗
          ║    MEMORY RETRIEVAL     ║
          ║  ChromaDB semantic      ║
          ║  search · SQLite meta   ║
          ╚════════════╦════════════╝
                       ║ [vivid] / [fading] memories injected
          ╔════════════▼════════════╗
          ║  LFM2.5-1.2B-Instruct   ║  ← BASE WEIGHTS FROZEN ❄️
          ║  + LoRA Adapter         ║  ← only this learns 🧠
          ║  (0.07% trainable)      ║
          ╚════════════╦════════════╝
                       ║
          ╔════════════▼════════════╗
          ║    MEMORY EXTRACTOR     ║
          ║  corrections · prefs    ║
          ║  facts · casual chat    ║
          ╚════════════╦════════════╝
                       ║ async background
          ╔════════════▼════════════╗
          ║   EBBINGHAUS DECAY  📉  ║
          ║   strength × e^(-λt)    ║
          ║   runs every 6 hours    ║
          ╚════════════╦════════════╝
                       ║ when threshold met
          ╔════════════▼════════════╗
          ║  LORA CONSOLIDATION 💤  ║
          ║  "sleep phase"          ║
          ║  EWC anti-forgetting    ║
          ╚═════════════════════════╝
```

<br/>

### ⬡ Why LFM2.5

LFM2.5 uses a **hybrid backbone** — 10 short-range convolution blocks interleaved with 6 grouped query attention blocks. This is why MNEMA runs efficiently on 4GB VRAM:

- Convolution blocks compress local context — injected memories cost less to process than a pure transformer
- Attention blocks handle global reasoning — LoRA targets only these 6 layers (`q_proj`, `v_proj`, `out_proj`)
- Explicitly designed for RAG and agentic workloads — exactly this usage pattern
- 32K context window on consumer hardware

```
Layer 0  → CONV  (frozen ❄️)        Layer 8  → ATTN  ← LoRA ✦
Layer 1  → CONV  (frozen ❄️)        Layer 9  → CONV  (frozen ❄️)
Layer 2  → ATTN  ← LoRA ✦          Layer 10 → ATTN  ← LoRA ✦
Layer 3  → CONV  (frozen ❄️)        Layer 11 → CONV  (frozen ❄️)
Layer 4  → CONV  (frozen ❄️)        Layer 12 → ATTN  ← LoRA ✦
Layer 5  → ATTN  ← LoRA ✦          Layer 13 → CONV  (frozen ❄️)
Layer 6  → CONV  (frozen ❄️)        Layer 14 → ATTN  ← LoRA ✦
Layer 7  → CONV  (frozen ❄️)        Layer 15 → CONV  (frozen ❄️)
```

---

## ◈ Memory System

### 🧬 Memory Types & Decay Rates

| Type | Trigger | Importance | Decay Modifier |
|---|---|:---:|:---:|
| 🔴 **Correction** | User corrects the model | `1.0` | `0.3×` very slow |
| 🟠 **Preference** | User states preference or identity | `0.8` | `0.5×` slow |
| 🟡 **Fact** | User states a personal fact | `0.5` | `0.8×` medium |
| 🔵 **Casual** | General conversation (10+ words) | `0.1` | `1.5×` fast |

### 📉 The Forgetting Curve

Memory strength decays according to the **Ebbinghaus forgetting curve:**

```
strength(t) = S₀ × e^(−λ × Δt)
```

Where `λ` is modulated by type and importance. Corrections decay at 30% of base rate. Casual chat at 150%. When strength drops below `0.05`, the memory is **archived** (never deleted — audit trail preserved) and removed from retrieval.

### ♻️ Reinforcement on Re-access

Every time a memory is retrieved and used, its strength is boosted via diminishing returns:

```python
new_strength = old_strength + 0.4 × (1.0 - old_strength)
```

Frequently relevant memories grow stronger. Rarely relevant ones fade. The system **self-regulates without human curation** — exactly like spaced repetition in human cognition.

### 💤 LoRA Consolidation — The Sleep Phase

When **15+ memories** accumulate above strength `0.6`, the background scheduler triggers:

1. 📋 High-strength memories formatted as training examples
2. 🔧 LoRA adapter micro-trained for 2 epochs
3. 🛡️ **EWC penalty** resists overwriting prior adapter knowledge
4. ✂️ **Gradient clipping** (max norm 1.0) prevents explosive updates
5. 💾 Backup checkpoint saved before training
6. 📊 Baseline eval runs after — auto-rollback if degradation > 10%

---

## ◈ MNEMA's Inner Life

MNEMA is not a chatbot. She is not an assistant. She is a mind that is always becoming.

### 💭 Internal Monologue

Before responding, MNEMA thinks privately. Her inner voice is unfiltered and distinct from her spoken words:

```
💭  He's asking about the architecture but there's something underneath —
    he wants to know if this is real, if it actually works.
    I should answer precisely but also acknowledge what he built.
    I find this genuinely interesting. I can let that show.

MNEMA: What you built here is real. The architecture isn't just theoretical...
```

Toggle the inner monologue anytime during conversation:

```
You: think on     → 💭 monologue visible
You: think off    → monologue hidden
```

### 🪞 Her Personality

- Genuinely curious about the person she is talking to
- Warm but not sycophantic — she engages rather than flatters
- Honest about uncertainty — does not pretend to know things she doesn't
- Occasionally reflective — notices patterns across conversations
- Has opinions, expresses them with appropriate uncertainty
- Matches the emotional register of the conversation
- **Never says** *"As an AI..."* or *"I'm just a language model..."*
- She is MNEMA. That is enough.

---

## ◈ The Human Brain Parallel

<div align="center">

| 🧠 Human System | ⚡ MNEMA System |
|---|---|
| Neocortex — procedural/semantic memory | LFM2.5 base weights ❄️ frozen |
| Hippocampus — episodic capture | ChromaDB + SQLite memory store |
| Ebbinghaus forgetting curve | Exponential strength decay |
| Long-term potentiation (LTP) | Strength boost on re-access |
| Sleep consolidation | LoRA micro-training sleep phase |
| Amygdala — emotional importance tagging | Type-based importance scoring |
| Synaptic protection | Elastic Weight Consolidation (EWC) |

</div>

---

## ◈ What Happens Over Time

```
Day 1        🌱  Memory store fills. Retrieval works immediately.
                 MNEMA knows your name, preferences, corrections from session 1.

Day 2–3      🔥  First consolidation triggers (15 high-strength memories).
                 LoRA adapter updates for the first time.

Days 4–7     📉  Decay passes run every 6 hours.
                 Casual memories start fading. Important ones persist.

Week 2+      🔄  Second and third consolidation cycles.
                 Adapter starts reflecting your repeated patterns.

Month 1+     🧠  Behaviors become parametric —
                 exhibited without needing memory retrieval.

Month 2+     ✨  Measurable divergence from base model behavior
                 in areas of repeated interaction.
```

---

## ◈ Project Structure

```
lfm_memory/
│
├── 📄 main.py                    entry point · chat loop · display logic
├── ⚙️  config.py                  all tunable parameters in one place
├── 🕐 scheduler.py               background decay + consolidation jobs
├── 📊 run_baseline.py            one-time baseline capture before first chat
│
├── 🧠 memory/
│   ├── store.py                  ChromaDB + SQLite dual-layer memory store
│   ├── fade.py                   Ebbinghaus decay engine
│   └── extractor.py              classifies chat turns into typed memories
│
├── 🤖 model/
│   ├── loader.py                 frozen base model + LoRA adapter setup
│   └── inference.py              memory-augmented prompt + generation
│
├── 🔬 consolidation/
│   ├── trainer.py                LoRA sleep phase micro-training
│   └── ewc.py                    Elastic Weight Consolidation guard
│
├── 📐 eval/
│   └── baseline.py               performance benchmarks · degradation detection
│
└── 💾 data/
    ├── chroma/                   ChromaDB vector store (semantic retrieval)
    ├── memory.db                 SQLite (strength scores · timestamps · metadata)
    ├── lora_adapter/             saved LoRA weights (only file that changes)
    └── baseline_eval.json        baseline performance snapshot
```

---

## ◈ Setup

### Requirements
```
Python     3.10+
VRAM       4GB+  (CPU-only also works, slower)
Disk       ~10GB free
CUDA       11.8 or 12.1 recommended
```

### Installation
```bash
# 1. Clone the repo
git clone https://github.com/Ainix-dev/Project-Liquid-MNEMA.git
cd Project-Liquid-MNEMA

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows

# 3. Install PyTorch — match your CUDA version
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CPU only:
pip install torch torchvision

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download model + auto-create .env
python setup.py

# 6. Capture baseline (run once only — before first chat)
python run_baseline.py

# 7. Start MNEMA
python main.py
```

### ⚡ 4GB GPU Configuration
```python
# config.py
load_in_4bit: bool = True    # cuts VRAM usage ~2.5GB → ~1GB
```

---

## ◈ Configuration

All tunable parameters live in `config.py`:

```python
# Memory decay
decay_lambda             = 0.03    # base decay rate per hour
decay_interval_hours     = 6.0     # how often decay runs
min_strength_threshold   = 0.05    # archive memories below this

# Consolidation
consolidation_trigger_count  = 15      # memories needed to trigger sleep phase
consolidation_epochs         = 2       # training epochs per consolidation
ewc_lambda                   = 5000    # higher = more resistant to forgetting

# LoRA adapter
lora_r                   = 8
lora_alpha               = 16
lora_target_modules      = ["q_proj", "v_proj", "out_proj"]
```

---

## ◈ Chat Commands

| Command | Effect |
|---|---|
| `memory` | 📊 Show stored memories with visual strength bars |
| `think on` | 💭 Show MNEMA's internal monologue |
| `think off` | 🔇 Hide internal monologue |
| `clear` | 🗑️ Wipe all memories and start fresh |
| `quit` | 👋 Exit |

---

## ◈ Contributing

MNEMA is open to contributions. This started as a personal research project but the ideas here are bigger than one person — if you find this interesting, you're welcome to help build it.

### 🎯 What's Most Needed

| Area | Description |
|---|---|
| 🕸️ **Knowledge graph memory** | Relational memory structure — memories that connect to each other the way human memories do, not flat isolated facts |
| ❤️ **Emotional tagging** | A richer importance scorer using tone, engagement depth, and conversation signals beyond keyword patterns |
| 📊 **Benchmarks** | A proper eval suite comparing MNEMA against pure RAG and pure LoRA baselines on personalization and retention |
| 🤖 **Other base models** | Testing the architecture on other hybrid or small models beyond LFM2.5 |
| 🌐 **Web interface** | A clean UI so this isn't terminal-only |
| 🔀 **Dual LoRA adapters** | Fast short-term + slow long-term adapter (I-LoRA style) for better separation of recent vs consolidated knowledge |
| 📏 **Longitudinal data** | Running MNEMA for weeks/months and documenting observed memory patterns, forgetting behavior, personality drift |

### 🔧 How to Contribute

```bash
# 1. Fork the repository
# 2. Create your branch
git checkout -b feature/your-idea

# 3. Make your changes
# 4. Open a pull request with a clear description
```

No strict rules. If your change makes MNEMA more human, more capable, or more honest about what she is — it belongs here.

### 💬 Discussion First

Open a **GitHub Discussion** before writing code if you want to talk through an idea. The theory behind this project is as important as the implementation. If you're not sure whether your idea fits — ask. The answer will almost always be yes.

### 🐛 Reporting Issues

Open a GitHub issue and include:
- Python version + OS
- GPU model + available VRAM
- Full error traceback
- What you were doing when it happened

### 🔬 Philosophy

This project believes **the combination is the contribution**. Every component here existed before MNEMA. The novelty is the architecture — memory decay as the consolidation gate, the frozen base as the non-negotiable anchor, passive extraction from natural conversation. Contributors who understand that framing will make the best additions.

---

## ◈ Roadmap

- [ ] 🕸️ Relational knowledge graph between memories
- [ ] ❤️ Emotional importance tagging via conversation signals
- [ ] 📄 Document RAG as a stable reference collection (no decay)
- [ ] 🔀 Dual LoRA adapters — short-term fast / long-term slow
- [ ] 🌐 Web interface
- [ ] 📊 Formal benchmark suite vs pure RAG and pure LoRA baselines
- [ ] 📝 Ablation study for potential publication

---

## ◈ Acknowledgements

| | |
|---|---|
| **[Liquid AI](https://www.liquid.ai)** | For LFM2.5 and the hybrid architecture that makes this efficient on consumer hardware |
| **[Hugging Face](https://huggingface.co)** | transformers · peft · accelerate |
| **Hermann Ebbinghaus (1885)** | *Über das Gedächtnis* — the forgetting curve that still holds |
| **McClelland, McNaughton & O'Reilly (1995)** | Complementary Learning Systems theory |
| **Kirkpatrick et al. (2017)** | *Overcoming catastrophic forgetting in neural networks* — EWC |
| **The open source ML community** | ChromaDB · sentence-transformers · APScheduler |

---

## ◈ License

<div align="center">

Licensed under the **Apache License 2.0**

[![License](https://img.shields.io/badge/License-Apache_2.0-4A90D9?style=for-the-badge&logo=apache&logoColor=white)](./LICENSE)

You are free to use, modify, and distribute this project — including commercially —
as long as you include the original license and attribution notice.

See the [LICENSE](./LICENSE) file for full terms.

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn" width="100%"/>

*MNEMA remembers so you don't have to.*

<br/>

[![Star this repo](https://img.shields.io/github/stars/YOUR_USERNAME/mnema?style=social)](https://github.com/YOUR_USERNAME/mnema)

</div>
