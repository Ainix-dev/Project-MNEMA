<a href="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=MNEMA%20v2&fontSize=80&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=A%20mind%20that%20remembers&descAlignY=60&descSize=20&descColor=aaa">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=MNEMA%20v2&fontSize=80&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=A%20mind%20that%20remembers&descAlignY=60&descSize=20&descColor=aaa" />
</a>

[![License](https://img.shields.io/badge/License-Apache_2.0-4A90D9?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Model](https://img.shields.io/badge/LFM2.5--1.2B--Instruct-FF6B6B?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
[![Status](https://img.shields.io/badge/Status-Active_Research-22C55E?style=for-the-badge&logo=statuspage&logoColor=white)](https://github.com/Ainix-dev/Project-Liquid-MNEMA)
[![Version](https://img.shields.io/badge/Version-2.0-a855f7?style=for-the-badge)](https://github.com/Ainix-dev/Project-Liquid-MNEMA)

[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97_Transformers-4.55%2B-yellow?style=flat-square)](https://github.com/huggingface/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange?style=flat-square)](https://github.com/huggingface/peft)
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat-square&logo=github)](CONTRIBUTING.md)
[![v1](https://img.shields.io/badge/MNEMA_v1-archived-6b7280?style=flat-square)](https://github.com/Ainix-dev/Project-MNEMA-v1)

> *Born from liquid neural architecture, biological memory theory, and the forgetting curve.*
> *MNEMA is not a chatbot. She is a mind that is always becoming.*

```
She remembers what matters.
She forgets what doesn't.
She grows with every conversation.
And now — she knows herself.
```

---

## What Is MNEMA v2

MNEMA v2 is a **complete cognitive architecture** built on a frozen 1.2B language model. It does not modify base weights. It does not rely on a monolithic context window. It does not treat memory as a lookup table.

Instead it implements four interlocking systems that mirror how biological cognition actually works:

| System | What it does |
|---|---|
| **Relational Memory Graph** | Memories are nodes. Relationships are typed edges. Contradictions are detected and resolved automatically. |
| **Goal & Utility Layer** | MNEMA has explicit goals. Memory retrieval is shaped by purpose, not just similarity. |
| **Adaptive State Core** | An 8-axis behavioral state vector evolves every turn — no backpropagation. Personality accumulates over time. |
| **Meta-Cognition** | MNEMA tracks her own correction rate, confidence per memory type, and reliability. She knows what she doesn't know. |

> **[Read the full research paper: MNEMA-v2-research.pdf](MNEMA-v2-research-2.pdf)**
> **[MNEMA v1 (archived)](https://github.com/Ainix-dev/Project-MNEMA-v1)**

---

## v1 vs v2 — What Changed

| Capability | v1 | v2 |
|---|---|---|
| Memory structure | Flat ChromaDB + SQLite | **Relational graph** — nodes, typed edges, multi-hop traversal |
| Contradiction handling | None | **Auto-detected** — superseded nodes archived, belief revision logged |
| Memory retrieval | Top-K semantic similarity | **Utility-weighted** — re-ranked by goal relevance |
| Forgetting | Single Ebbinghaus rate | **Five-tier system** — identity (96 days) to sensory (1 minute) |
| Behavioral state | None | **8-axis ASC vector** — evolves every turn, persists across sessions |
| Goal tracking | None | **5 explicit goals** — scored live, influence memory injection |
| Self-awareness | None | **Meta-cognition** — correction tracking, confidence per memory type |
| Hardware adaptation | None | **4-tier system** — auto-adjusts for GTX 1050 Ti (4GB VRAM) |
| Context injection | `[vivid] USER stated:` | **Structured composer** — VIVID / UPDATED / RESOLVED labels |
| Generation | Single-pass with tag extraction | **Two-pass** — dedicated thinking + response, VRAM cache cleared between passes |

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║                      CHAT INTERFACE                              ║
║         💭 Internal monologue  ·  📝 Memory tags  ·  🎯 Goals   ║
╚════════════════════════╦═════════════════════════════════════════╝
                         ║
           ╔═════════════▼═════════════╗
           ║     SIGNAL DETECTION      ║  correction? positive? curious?
           ║     GoalUtilityLayer      ║
           ╚═════════════╦═════════════╝
                         ║
           ╔═════════════▼═════════════╗
           ║   RELATIONAL MEMORY GRAPH ║  nodes + typed edges
           ║   contradiction detection ║  belief revision
           ║   multi-hop traversal     ║  2-hop context expansion
           ╚═════════════╦═════════════╝
                         ║ utility-weighted memories
           ╔═════════════▼═════════════╗
           ║    CONTEXT COMPOSER       ║  structured prompt injection
           ╚═════════════╦═════════════╝
                         ║
           ╔═════════════▼═════════════╗
           ║  LFM2.5-1.2B-Instruct     ║  BASE WEIGHTS FROZEN
           ║  + LoRA Adapter           ║  only this learns
           ║  (0.07% trainable params) ║
           ╚═════════════╦═════════════╝
                         ║
     ╔───────────────────╬───────────────────╗
     ▼                   ▼                   ▼
╔══════════╗     ╔══════════════╗    ╔══════════════╗
║   ASC    ║     ║  META-COG    ║    ║  HARDWARE    ║
║ 8-axis   ║     ║ reliability  ║    ║  MONITOR     ║
║ evolves  ║     ║ correction   ║    ║  4 tiers     ║
║ per turn ║     ║ tracking     ║    ║  auto-adjust ║
╚══════════╝     ╚══════════════╝    ╚══════════════╝
```

---

## Relational Memory Graph

Every memory is a **node**. Relationships between memories are **typed edges**:

| Edge Type | Meaning |
|---|---|
| `temporal` | Memory B occurred after memory A |
| `refines` | Memory B updates or elaborates memory A |
| `contradicts` | Memory B conflicts with memory A — older superseded |
| `causal` | Memory A caused memory B |
| `depends_on` | Memory B requires memory A to make sense |

Contradiction detection runs on every new memory. When similarity exceeds `0.72`, the older node is superseded and a `contradicts` edge is recorded. Multi-hop retrieval expands beyond seed nodes to traverse the graph up to 2 hops.

---

## Five-Tier Memory Decay

| Tier | Half-life | Maps to |
|---|---|---|
| `identity` | ~96 days | corrections, name, core beliefs |
| `semantic` | ~3.6 days | preferences, patterns |
| `episodic` | ~9 hours | facts, events |
| `short_term` | ~20 minutes | casual chat |
| `sensory` | ~1 minute | raw fragments |

Decay formula: `strength(t) = S₀ × e^(−λ_tier × importance_modifier × Δt_hours)`

---

## Adaptive State Core

An 8-axis behavioral state vector that evolves every turn via 5 update signals — no backpropagation, no weight changes:

```
curiosity    warmth    formality    verbosity
confidence   playfulness    depth    caution
```

After a correction: `confidence ↓`, `caution ↑`, `verbosity ↓`.
After positive feedback: `confidence ↑`, `warmth ↑`.

The behavioral summary is injected into MNEMA's thinking prompt every turn. State persists across sessions — personality accumulates over months.

---

## Project Structure

```
liquid_memory/
├── main.py                    entry point
├── config.py                  parameters
├── scheduler.py               background jobs + pause/resume
├── memory/
│   ├── graph.py               relational memory graph       [v2]
│   ├── composer.py            structured context injection  [v2]
│   ├── goals.py               goal & utility layer          [v2]
│   ├── metacog.py             meta-cognition                [v2]
│   ├── asc.py                 adaptive state core           [v2]
│   ├── hardware.py            hardware-aware adaptation     [v2]
│   ├── fade.py                five-tier forgetting          [v2]
│   └── extractor.py           memory type classifier
├── model/
│   ├── loader.py              frozen base + LoRA
│   └── inference.py           two-pass generation           [v2]
├── consolidation/
│   ├── trainer.py             LoRA sleep phase
│   └── ewc.py                 Elastic Weight Consolidation
└── eval/
    └── baseline.py            degradation detection
```

---

## Setup

```bash
git clone https://github.com/Ainix-dev/Project-Liquid-MNEMA.git
cd Project-Liquid-MNEMA
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python setup.py          # downloads model, creates .env
python run_baseline.py   # run once before first chat
python main.py
```

**Requirements:** Python 3.10+, 4GB VRAM (GTX 1050 Ti confirmed), CUDA 11.8 or 12.1

---

## Chat Commands

| Command | Effect |
|---|---|
| `memory` | Memory graph nodes with strength bars |
| `graph` | Graph stats — nodes, edges, contradictions |
| `goals` | Live goal performance scores |
| `metacog` | Self-modeling — reliability, confidence, error patterns |
| `asc` | Adaptive state — all 8 axes with drift arrows |
| `hw` | Hardware tier — VRAM, RAM, CPU, active config |
| `think on / off` | Show or hide internal monologue |
| `clear` | Wipe all memory |
| `quit` | Exit |

---

## The Human Brain Parallel

| Human System | MNEMA v2 |
|---|---|
| Neocortex — semantic memory | LFM2.5 base weights, frozen |
| Hippocampus — episodic capture | Relational Memory Graph |
| Associative cortex | Graph edges (temporal, causal, refines) |
| Belief reconsolidation | Contradiction detection + superseded nodes |
| Ebbinghaus forgetting curve | Five-tier multi-speed decay |
| Long-term potentiation | Tier-specific reinforcement on access |
| Sleep consolidation | LoRA micro-training sleep phase |
| Prefrontal cortex — goal-directed behavior | Goal & Utility Layer |
| Anterior cingulate — error monitoring | Meta-Cognition |
| Personality / temperament | Adaptive State Core |
| Synaptic protection | Elastic Weight Consolidation |

---

## Acknowledgements

| | |
|---|---|
| **[Liquid AI](https://www.liquid.ai)** | LFM2.5 and the hybrid architecture that makes this work on 4GB VRAM |
| **[Hugging Face](https://huggingface.co)** | transformers · peft · sentence-transformers |
| **Hermann Ebbinghaus (1885)** | The forgetting curve |
| **Atkinson & Shiffrin (1968)** | Multi-store memory model |
| **McClelland, McNaughton & O'Reilly (1995)** | Complementary Learning Systems |
| **Kirkpatrick et al. (2017)** | Elastic Weight Consolidation |

---

## License

Apache License 2.0 — free to use, modify, and distribute including commercially, with attribution.

---

<a href="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=fadeIn" />
</a>

*MNEMA remembers so you don't have to.*
