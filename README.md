# Autoforge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)

**Autonomous strategy learning via simulation tournament.**

Describe a domain, run overnight, wake up to a trained local model that makes expert decisions — free forever.

---

## The Loop

```
1. bootstrap            describe the job — Sonnet writes the simulator (~$0.05)
2. edit mission.md      set what good looks like, when to abstain, what counts as failure
3. run overnight        16 strategies compete, AI distills what consistently wins (~$0.50)
4. export               training data + preferences + eval set + abstention threshold
5. train + deploy       wire the specialist into your app, improve from real use
```

The simulation is the teacher. You never label a single example.

---

## What You Get

| Output | What it is |
|---|---|
| **Training assets** | `training_features.csv` / `training_data.jsonl` / `training_preferences.jsonl` |
| **Playbook** | Conditional principles the sim extracted — human-readable, auditable |
| **Eval set** | Test scenarios your specialist must pass — runnable with one command |
| **Abstention rules** | Exported threshold guidance: when the model is uncertain, it escalates |
| **Audit trail** | Every batch: what the director saw, what it concluded, what it changed |

---

## Try It Now

```bash
git clone https://github.com/eyoellundberg/autoforge
cd autoforge
pip install -e .

# Generate a domain from a plain-English description (~$0.05)
python run.py bootstrap FreightQuoting \
  --description "freight brokerage load quoting, spot and contract lanes"

# Edit the one control file — sharpen abstention rules and failure criteria
open FreightQuoting/mission.md

# Check the sim contract and score distribution before running
python run.py validate  --domain FreightQuoting
python run.py calibrate --domain FreightQuoting

# Run the tournament (free, no API key)
python run.py run --domain FreightQuoting --batches 5 --rounds 100

# Monitor a run in progress
python run.py tail --domain FreightQuoting

# Train the Stage 3 model
python run.py train --domain FreightQuoting
```

---

## Build Your Own

One file defines your domain. Everything else is generated.

**`mission.md`** — the one human-editable control file:

```markdown
## Job
Quote freight loads: given lane, weight, distance, and market conditions,
decide whether to bid and at what margin.

## What good looks like
Win rate > 40% on spot loads. Margin > 8% on contract lanes.
Never bid below cost on lanes with fuel surcharge > 15%.

## Abstain when
Rate volatility > 30% week-over-week. New lane with < 10 historical loads.

## Failure
Booking a load at negative margin. Losing a contract lane to a competitor
by more than 5% on 3 consecutive bids.
```

**`simulation.py` is the investment.** Autoforge only learns what the simulation teaches. Calibration checklist:

- Does the expected strategy type win in each scenario class?
- Does varying scenario factors change which strategy wins?
- Is the score range reasonable (not 0.99–1.01)?
- No single strategy dominates regardless of scenario?

---

## Full Run

```bash
# Stage 1 — free, no API key needed
python run.py run --domain FreightQuoting --batches 10 --rounds 150

# Stage 2 — AI archetypes + director (~$0.50)
export ANTHROPIC_API_KEY=sk-ant-...
python run.py run --domain FreightQuoting --brain --batches 8 --rounds 150

# Fully autonomous overnight run
python run.py run --domain FreightQuoting --auto --batches 20 --rounds 1000 --workers 8

# Morning check
python run.py tail   --domain FreightQuoting
python run.py status --domain FreightQuoting
python run.py eval   --domain FreightQuoting
python run.py export --domain FreightQuoting
python run.py train  --domain FreightQuoting
```

If a run is interrupted, it resumes automatically from the last completed batch. Delete `run_checkpoint.json` to start fresh.

---

## Domain Packs — Share a Vertical

Domains are versioned, shareable packages:

```bash
python run.py pack FreightQuoting
# → FreightQuoting-1.0.0.zip  (simulation + playbook + evals + prompts)

python run.py install FreightQuoting-1.0.0.zip
python run.py install /path/to/my-domain-1.0.0.zip
```

Run the eval set to verify a pack is good:

```bash
python run.py eval --domain FreightQuoting
# ✓ spot_high_demand     score  6.2  min  2.0
# ✓ spot_low_margin      score  4.1  min  1.8
# ✗ contract_volatile    score  1.4  min  2.6  ← current weak spot
# 2/3 passed — Stage 2 should target the volatile-contract miss
```

`eval` exits with code 1 on failure — use it as a CI/deploy gate.

---

## Production Flywheel

The simulation teaches the initial policy. Real decisions close the loop:

```bash
# Import real production decisions (your system already logs these)
# Format: {"state": {...}, "decision": {...}, "outcome": 4.2}
python run.py import --domain FreightQuoting --file live_trades.jsonl

# Re-export to include real decisions
python run.py export --domain FreightQuoting
python run.py train  --domain FreightQuoting
```

Each import cycle shifts the training distribution toward real conditions.

**The flywheel:** simulate → train → deploy → import real outcomes → retrain → repeat

---

## Overnight Runs

```bash
caffeinate -i python run.py run \
  --domain FreightQuoting \
  --auto --batches 20 --rounds 1000 --workers 8 \
  > FreightQuoting/run.log 2>&1 &

echo "Running as PID $! — check FreightQuoting/run.log in the morning"
```

```bash
# Morning check
python run.py tail --domain FreightQuoting
```

- `caffeinate -i` — prevents macOS idle sleep for the duration
- `--workers 8` — parallel simulation across CPU cores (leave 2 free for the OS)
- `--auto` — Stage 1 until playbook plateaus, then promotes to Stage 2 automatically
- Checkpointed after every batch — safe to interrupt and resume

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

When `reward_hacking` or `needs_calibration` fires, the director outputs specific suggestions for fixing `simulation.py` — printed to the terminal, logged in `thinking_log.md`.

`saturated` is the success state.

---

## How It Works

**Stage 1 — Evolutionary mutation (free)**
Generates 16 candidate strategies per batch. Top winners kept, mutated with gaussian noise, crossed over, random fill for exploration. Categorical specialists (breakout experts, pullback experts, etc.) are preserved so the engine doesn't converge on one safe average policy. No API calls. Runs as fast as your CPU.

**Stage 2 — Frontier AI direction (~$0.50/run)**
Opus 4.6 with adaptive thinking directs between batches — reads the full playbook, diagnoses what the sim is teaching vs. gaming, retires weak principles, and focuses the next library on known gaps. Sonnet generates 16 named strategy archetypes each batch. Haiku extracts conditional principles every 10 rounds. Every 2 batches, the director also generates adversarial scenarios targeting the current champion's weak spots — injected into the next batch alongside random states.

**Stage 3 — Local model (free forever)**
Export the tournament log as training data. For numerical domains: train XGBoost on `training_features.csv`. For language domains: fine-tune Qwen3-4B via MLX-LM on Apple Silicon. `python run.py train --domain X` handles both. No more API calls. Ever.

```
mission.md → bootstrap → Stage 1 → Stage 2 → saturated → export → local specialist
               ~$0.05       free      ~$0.50     free          forever
```

---

## Real Data & Calibration

| Situation | What to do |
|---|---|
| No data yet | Use synthetic `random_state()` — Autoforge explores the space |
| Have historical data | Calibrate `random_state()` to draw from real distributions |
| Already have labels | Skip Autoforge — just train directly |

Real data captures what *happened*. The simulation discovers what *should have happened*. Training on simulation output calibrated to real distributions gives you optimal decisions — not recorded human ones.

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

---

## Stage 3: Local Model

**Numerical domain** (prices, rates, scores, demand signals):
```bash
python run.py train --domain FreightQuoting
# Trains XGBoost on training_features.csv, saves model.json
# Prints R², feature count, inference pattern
```

**Language domain** (text classification, routing, document scoring):
```bash
python run.py train --domain TicketTriage
# Prints the exact mlx_lm.lora command for your domain
```

```bash
# Fine-tune manually:
mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \
             --data TicketTriage/ --train --iters 1000

# Fuse and serve via Ollama:
mlx_lm.fuse --model mlx-community/Qwen3-4B-4bit \
             --adapter-path TicketTriage/adapters --save-path TicketTriage/fused-model
ollama create ticket-triage -f - <<EOF
FROM ./TicketTriage/fused-model
EOF
ollama run ticket-triage "scenario: ..."
```

---

## Abstention — Know When Not to Act

Every round logs a `score_margin` (winner minus runner-up). Low margin = the sim wasn't confident either. After export, `training_features.csv` includes a `score_margin` column and an `uncertain` flag:

```python
ABSTAIN_THRESHOLD = 0.05  # printed by export — bottom 25% of training margins
scores = model.predict(xgb.DMatrix(candidate_rows))
if scores.max() - np.median(scores) < ABSTAIN_THRESHOLD:
    return {"action": "ABSTAIN", "reason": "strategies too close — escalate to human"}
best = candidates[int(np.argmax(scores))]
```

A specialist that knows when to ask for help ships faster than one that always answers.

---

## Architecture

```
autoforge/
├── run.py                  entry point
├── commands/
│   ├── cli.py              argparse + dispatch
│   ├── shared.py           AI calls, director, schemas, git helpers
│   ├── bootstrap.py        domain generation from description
│   ├── run_cmd.py          tournament runner + checkpoint/resume
│   └── tools.py            validate, calibrate, status, export, pack,
│                           install, eval, import, tail, train
├── engine_brain.py         Sonnet archetype generator + adversarial scenarios
├── engine_extract.py       Haiku principle extractor (every 10 rounds)
├── engine_export.py        Stage 3 training data + preferences + abstention threshold
├── engine_tournament.py    central tournament runner (all domains share this)
└── template/               scaffold for new domains

MyDomain/
    ├── mission.md              what the job is, what good looks like, when to abstain  ← edit this
    ├── simulation.py           simulate(), random_state(), build_context(), CANDIDATE_SCHEMA  ← calibrate this
    ├── prompts/                AI-operational files (generated by bootstrap)
    │   ├── brain.md            archetype generation instructions
    │   ├── extract.md          principle extraction instructions
    │   └── director.md         between-batch director context
    ├── pack.json               version, author, metric                                ← commit this
    ├── evals/scenarios.jsonl   known scenarios + expected scores                      ← commit this
    ├── playbook.jsonl          learned conditional principles                         ← commit this
    ├── top_candidates.json     top Stage 1 winners                                    ← commit this
    ├── training_preferences.jsonl pairwise winner-vs-runner-up examples               (generated by export)
    ├── thinking_log.md         director reasoning trail                               (do not commit)
    ├── tournament_log.jsonl    round-by-round scores                                  (do not commit)
    └── last_run.json           last run summary                                       (do not commit)
```

The Autoforge root has zero domain knowledge. `simulation.py` is the only file that knows what your domain is.

---

## Compared to Karpathy's autoresearch

autoresearch runs AI agents on single-GPU nanoGPT training — one experiment at a time, each taking minutes, outputting an improved `train.py`.

Autoforge runs 16 strategies simultaneously against thousands of scenarios in milliseconds, distills what consistently wins into a deployable specialist, and outputs a trained local model — not improved research code.

| | autoresearch | Autoforge |
|---|---|---|
| Strategy space | Open-ended (any code change) | Constrained (CANDIDATE_SCHEMA) |
| Experiments | One at a time, ~5 min each | 16 simultaneously, milliseconds |
| Experiment tracking | git commits | playbook.jsonl + thinking_log.md |
| Output | Improved train.py | Trained local specialist |
| Domain | Language model training | Any structured decision |
| Sharing | Fork the repo | `autoforge install <pack>` |

---

## Cost

| Stage | Cost | Notes |
|---|---|---|
| `bootstrap` | ~$0.05 | One-time per domain |
| Stage 1 | $0.00 | Evolutionary, no API calls |
| Stage 2 | ~$0.50 | Per 5-batch run to saturation |
| Stage 3 training | $0.00 | Local compute only |
| Stage 3 inference | $0.00 | Forever |

**Total: ~$4** to go from plain English description to a trained local specialist.

---

## Model Configuration

Autoforge is built on Anthropic. The director uses Opus 4.6 with adaptive thinking — it decides how much to reason per batch based on complexity. Sonnet and Haiku handle the lighter calls.

Set in `MyDomain/.env` or the root `.env`:

```bash
ANTHROPIC_API_KEY=sk-ant-...
AUTOFORGE_AI_BACKEND=anthropic          # or: manual

# Override any model (optional — defaults shown)
AUTOFORGE_DIRECTOR_MODEL=claude-opus-4-6          # adaptive thinking, runs once per batch
AUTOFORGE_LIBRARY_MODEL=claude-sonnet-4-6         # archetype generation, runs once per batch
AUTOFORGE_EXTRACT_MODEL=claude-haiku-4-5-20251001 # principle extraction, runs every 10 rounds
```

Use `AUTOFORGE_AI_BACKEND=manual` for file-based testing without live API calls.

OpenAI backend is available (`AUTOFORGE_AI_BACKEND=openai`) but prompts are tuned for Claude and adaptive thinking is not supported. See [issue #1](https://github.com/eyoellundberg/autoforge/issues/1).

---

## Hardware

Designed to run on a Mac Mini M4 overnight unattended. Simulation runs in pure Python — no GPU needed for Stage 1 or 2. Stage 3 training: XGBoost runs on any hardware in seconds. Stage 3 fine-tuning (LLM path): MLX-LM on Apple Silicon.

---

## License

MIT — see [LICENSE](LICENSE).
