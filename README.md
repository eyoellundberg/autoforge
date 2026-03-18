# Autoforge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)

**Autonomous strategy learning via simulation tournament.** Describe a decision domain, run overnight, wake up to a trained local model that makes expert decisions for free — forever.

---

## Why Autoforge?

Most decisions follow patterns that can be learned — but the training data doesn't exist yet. Autoforge creates it.

You describe the problem. It simulates thousands of scenarios, runs competing strategies against them, extracts what consistently wins, and distills that into a local model that costs nothing to run. No labelled dataset required. No ongoing API bill. **The simulation is the teacher.**

---

## How It Works

Three stages, each building on the last:

**Stage 1 — Evolutionary mutation (free)**
Generates 16 candidate strategies per batch. Each batch evolves from the last: top winners kept, mutated with gaussian noise, crossed over, random fill for exploration. No API calls. Runs as fast as your CPU.

**Stage 2 — Frontier AI direction (~$0.50/run)**
Sonnet reads the warm playbook and generates 16 named strategy archetypes — each with a philosophy, not just parameters. The sim tests all 16 across hundreds of scenarios. Haiku extracts conditional principles every 10 rounds. The director reads results between batches, retires losers, sharpens the next library. By batch 5, Sonnet is refining its own prior designs based on what the sim proved.

**Stage 3 — Local model (free forever)**
Export the tournament log as training data. For numerical domains: train XGBoost — tiny, fast, explainable, microsecond inference. For language domains: fine-tune Qwen 1.5B via MLX-LM on Apple Silicon. No more API calls. Ever.

```
Description → bootstrap → Stage 1 → Stage 2 → saturated → export → local model
    ~$0.05        free      ~$0.50     free        forever
```

---

## Two Filters

If both are true, Autoforge can learn it:

1. **Can the job be described in a markdown file?** If someone can write what good output looks like and what the inputs are — Autoforge can learn it.
2. **Is the output structured?** Pricing, scoring, triage, routing, forecasting, game AI — any job that maps inputs to a structured decision.

---

## Quickstart

```bash
git clone https://github.com/eyoellundberg/autoforge
cd autoforge
pip install -e .
```

```bash
# 1. Generate a domain from a plain English description
python run.py bootstrap GrainMarketing \
  --description "corn/soy marketing for midwest US farms, optimizing \
  timing of cash grain sales against local elevator basis"

# 2. Review GrainMarketing/simulation.py — the only manual step
#    Calibrate until the right strategy wins in each scenario type

# 3. Check distributions and catch issues before a long run
python run.py calibrate --domain GrainMarketing

# 4. Stage 1 — free, no API key needed
python run.py run --domain GrainMarketing --batches 10 --rounds 150

# 5. Stage 2 — AI archetypes + director
export ANTHROPIC_API_KEY=sk-ant-...
python run.py run --domain GrainMarketing --brain --batches 8 --rounds 150

# 6. Fully autonomous overnight run
python run.py run --domain GrainMarketing --auto --batches 20 --rounds 1000 --workers 8

# 7. Check what it learned
python run.py status --domain GrainMarketing

# 8. Export Stage 3 training data
python run.py export --domain GrainMarketing
```

---

## Try It Now — StockTiming Example

A complete working domain is included. No setup needed:

```bash
# Stage 1 — free, no API key
python run.py run --domain StockTiming --batches 5 --rounds 100

# Calibrate first to see scenario distributions
python run.py calibrate --domain StockTiming

# Stage 2 — AI archetypes (needs API key)
python run.py run --domain StockTiming --brain --batches 5 --rounds 100

# Check what it learned
python run.py status --domain StockTiming
```

StockTiming optimizes moving average crossover parameters across simulated market regimes (trending, ranging, volatile, event). Minimal but complete — use it to understand the loop before building your own domain.

---

## Adding Your Domain

Four things to provide. Everything else is Autoforge.

```
MyDomain/
├── simulation.py      # the scoring function — the only hard part
│                      # exports: simulate(), random_state(),
│                      #          CANDIDATE_SCHEMA, METRIC_NAME
├── tournament.py      # copy from template/, adapt _build_context()
└── prompts/
    ├── brain.md       # tell Sonnet what 16 archetypes to generate
    ├── extract.md     # tell Haiku what principles to extract
    └── director.md    # give the director domain context
```

**`simulation.py` is the investment.** Autoforge only learns what the simulation teaches. Calibration checklist:

- Does the expected strategy type win in each scenario class?
- Does varying scenario factors change which strategy wins?
- Is the score range reasonable?
- No single strategy dominates regardless of scenario?

Or skip the blank page — `bootstrap` generates all four files from a description:

```bash
python run.py bootstrap MyDomain --description "..."
```

---

## Real Data & Calibration

| Situation | What to do |
|---|---|
| No data yet | Use synthetic `random_state()` — Autoforge explores the space |
| Have historical data | Calibrate `random_state()` to draw from real distributions |
| Already have labels | Skip Autoforge — just train directly |

The most powerful case is the middle one. Real data captures what *happened*. The simulation discovers what *should have happened*. Training on simulation output calibrated to real distributions gives you optimal decisions — not recorded human ones.

```python
# simulation.py — load a CSV once, sample from real history
import csv, random
from pathlib import Path

_DATA = None

def _load_data():
    global _DATA
    if _DATA is None:
        with open(Path(__file__).parent / "data" / "history.csv") as f:
            _DATA = list(csv.DictReader(f))
    return _DATA

def random_state() -> dict:
    row = random.choice(_load_data())
    return {"demand": float(row["demand"]), "basis": float(row["basis"])}
```

Verify before a long run:

```bash
python run.py calibrate --domain GrainMarketing
# Shows: scenario distributions, score range, candidate win spread
# Flags: dominant strategies, zero-heavy scores, low variance
```

---

## Overnight Runs

```bash
# Run unattended — survives terminal close, prevents macOS sleep
caffeinate -i uv run python run.py run \
  --domain GrainMarketing \
  --auto --batches 20 --rounds 1000 --workers 8 \
  > GrainMarketing/run.log 2>&1 &

echo "Running as PID $! — check GrainMarketing/run.log in the morning"
```

```bash
# Morning check
tail -50 GrainMarketing/run.log
python run.py status --domain GrainMarketing
```

- `caffeinate -i` — prevents macOS idle sleep for the duration
- `--workers 8` — parallel simulation across CPU cores (leave 2 free for the OS)
- `--auto` — Stage 1 until playbook plateaus, then promotes to Stage 2 automatically
- Batch and director failures are caught and logged — a single API timeout won't kill the run

---

## Architecture

```
autoforge/
├── run.py                  entry point (8 lines)
├── commands/
│   ├── cli.py              argparse + dispatch
│   ├── shared.py           schemas, director AI, git helpers
│   ├── bootstrap.py        domain generation from description
│   ├── run_cmd.py          tournament runner
│   └── tools.py            calibrate, validate, status, export
├── engine_brain.py         Sonnet archetype library generator
├── engine_extract.py       Haiku principle extractor (every 10 rounds)
├── engine_export.py        Stage 3 training data exporter
└── template/               copy this to start a new domain
    ├── simulation.py       skeleton with calibration guide
    ├── tournament.py       tournament loop with evolution built in
    └── prompts/            brain.md / extract.md / director.md

MyDomain/
    ├── simulation.py           scoring function — only file with domain knowledge
    ├── tournament.py           tournament loop (template + _build_context)
    ├── prompts/                domain-specific AI instructions
    ├── data/                   optional: real historical data
    ├── playbook.jsonl          learned conditional principles  ← commit this
    ├── retired_topics.json     permanent principle blocklist   ← commit this
    ├── champion_archetype.json best archetype from last batch  ← commit this
    ├── top_candidates.json     top Stage 1 winners             ← commit this
    ├── thinking_log.md         director reasoning trail        (do not commit)
    ├── tournament_log.jsonl    round-by-round scores           (do not commit)
    └── last_run.json           last run summary                (do not commit)
```

The Autoforge root has zero domain knowledge. `simulation.py` is the only file that knows what your domain is.

---

## Convergence & Stopping

The director issues a verdict after every batch:

| Verdict | Meaning | Action |
|---|---|---|
| `converging` | Score and playbook improving steadily | Keep running |
| `exploring` | Mixed results, still searching | Keep running |
| `stalled` | No improvement for multiple batches | Adjust sim or prompts |
| `reward_hacking` | Score rising for wrong reason | Stop — fix simulation.py |
| `needs_calibration` | Sim rewarding wrong behavior | Fix simulation.py |
| `saturated` | Playbook full, score stable | Export and train local model |

`saturated` is the success state. Autoforge has learned everything this simulation can teach.

---

## Stage 3: Choosing a Local Model

**Numerical domain** (prices, rates, scores, demand signals):
```python
# export produces training_features.csv + labels
python run.py export --domain GrainMarketing
# train XGBoost — <1MB model, microsecond inference
import xgboost as xgb
model = xgb.train(params, xgb.DMatrix(X, label=y))
```

**Language domain** (text classification, routing, document scoring):
```bash
# export produces messages JSONL
python run.py export --domain TicketTriage
# fine-tune Qwen 1.5B via MLX-LM on Apple Silicon
mlx_lm.lora --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
             --data TicketTriage/ --train --iters 1000
```

XGBoost trained on 1,000 tournament examples will outperform a raw LLM on numerical decisions. It is the right tool for a regression problem. Qwen earns its place when the inputs contain language that needs understanding.

---

## Cost

| Stage | Cost | Notes |
|---|---|---|
| `bootstrap` | ~$0.05 | One-time per domain |
| Stage 1 | $0.00 | Evolutionary, no API calls |
| Stage 2 | ~$0.50 | Per 5-batch run to saturation |
| Stage 3 training | $0.00 | Local compute only |
| Stage 3 inference | $0.00 | Forever |

**Total: ~$4** to go from plain English description to a trained local model.

---

## Model Configuration

All models are configurable. Set in `MyDomain/.env` or the root `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Override any model (optional — defaults shown)
AUTOFORGE_DIRECTOR_MODEL=claude-sonnet-4-6        # between-batch director
AUTOFORGE_LIBRARY_MODEL=claude-sonnet-4-6         # archetype generation
AUTOFORGE_EXTRACT_MODEL=claude-haiku-4-5-20251001 # principle extraction
```

Tighter budget: swap Sonnet for Haiku everywhere. More quality: use Opus for the director.

---

## Compared to Karpathy's autoresearch

| | autoresearch | Autoforge |
|---|---|---|
| Strategy space | Open-ended (any code change) | Constrained (CANDIDATE_SCHEMA) |
| Experiments | One at a time, ~5 min each | 16 simultaneously, milliseconds |
| Experiment tracking | git commits | playbook.jsonl + thinking_log.md |
| Output | Improved train.py | Trained local model |
| Domain | Language model training | Any structured decision |

autoresearch goes deep on one idea at a time. Autoforge goes wide across many simultaneously, then distills what won into a deployable model.

---

## Hardware

Designed to run on a Mac Mini M4 overnight unattended. Simulation runs in pure Python — no GPU needed for Stage 1 or 2. Stage 3 training: XGBoost runs on any hardware in seconds. Stage 3 fine-tuning (LLM path): MLX-LM on Apple Silicon.

---

## License

MIT — see [LICENSE](LICENSE).
