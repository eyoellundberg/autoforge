# Autoforge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)

**Run millions of simulated scenarios to train a local specialist — before you have a single labeled example.**

You describe a decision your system needs to make. Autoforge writes a simulator, runs hundreds of thousands of competing strategies against it, and distills what consistently wins into a trained local model. No labeled data. No ongoing API costs after training.

---

## The Core Idea

Most ML projects stall waiting for labeled data. Autoforge sidesteps this entirely: **write a simulator that scores decisions, and the simulator becomes your dataset.**

The engine runs 16 competing strategies across hundreds of thousands of scenarios, extracts the conditional principles that separate winners from losers, and exports that as training data for a local model. A single overnight run produces enough signal to train a specialist that runs forever at zero cost.

---

## What You Can Train It On

Anything where you can write a function that takes a scenario and a candidate decision, and returns a score. The engine has no domain knowledge — it only knows how to run tournaments and extract what wins.

**Decision-making under uncertainty:**
- Quote a freight load at the right margin
- Enter a stock position given technical signals
- Route a support ticket to the right team
- Accept or reject a loan application
- Bid on a programmatic ad impression
- Triage a patient for urgency level

**Optimization with a clear objective:**
- Pricing strategy across customer segments
- Inventory reorder timing
- Content recommendation ranking
- Resource allocation across queues

**Policy learning where data is expensive or slow:**
- Clinical decision support (simulate patient outcomes)
- Legal document routing (simulate routing accuracy)
- Supply chain response to disruption scenarios
- Hiring screening criteria

If you can score a decision against a scenario, you can train a specialist.

---

## How It Works

```
1. bootstrap        describe your domain in plain English — AI writes the simulator (~$0.05)
2. edit mission.md  set what good looks like, when to abstain, what counts as failure
3. run overnight    16 strategies compete across hundreds of thousands of scenarios (~$0.50)
4. export           training data, eval set, abstention threshold
5. train + deploy   local specialist, runs forever at zero cost
```

The simulation is the teacher. You never label a single example.

---

## Try It Now

```bash
git clone https://github.com/eyoellundberg/autoforge
cd autoforge
pip install -e .                          # zero hard dependencies — pure Python + stdlib

# Generate a domain from a plain-English description (~$0.05)
python run.py bootstrap FreightQuoting \
  --description "freight brokerage load quoting, spot and contract lanes"

# Edit the one control file — set what winning looks like
open FreightQuoting/mission.md

# Check the simulator is healthy before running
python run.py validate  --domain FreightQuoting
python run.py calibrate --domain FreightQuoting

# Run the tournament (free, no API key needed)
python run.py run --domain FreightQuoting --batches 5 --rounds 100

# Monitor a live run
python run.py tail --domain FreightQuoting

# Train the local model
python run.py train --domain FreightQuoting
```

---

## The One File You Edit

**`mission.md`** defines the job. Everything else is generated.

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

**`simulation.py` is the investment.** The AI can only learn what the simulator teaches. Before running overnight, verify:

- The right strategy type wins in each scenario class
- Varying scenario factors changes which strategy wins
- Score range is meaningful (not 0.99–1.01)
- No single strategy dominates regardless of scenario

---

## The Simulator Contract

`simulation.py` is the only file that knows what your domain is. The engine is completely agnostic — it calls these four things:

```python
# Required
CANDIDATE_SCHEMA: dict         # JSON schema for a strategy (5–15 parameters)
METRIC_NAME: str               # e.g. "profit", "entry_utility", "routing_accuracy"

def random_state() -> dict:
    # Draw one scenario. Called once per round. Pure Python, no I/O.
    # Must cover diverse situations — this is your training distribution.

def simulate(candidate: dict, state: dict) -> float:
    # Score a candidate strategy against a scenario. Higher = better.
    # Must be deterministic. Called millions of times — keep it fast.

# Optional
def build_context(state: dict) -> dict:
    # Human-readable labels for principle extraction (e.g. "risk_bucket": "high")
```

A strategy is a JSON dict matching `CANDIDATE_SCHEMA`. The tournament discovers which parameterizations win across your scenario distribution — that signal becomes your training data.

---

## Scale

A single overnight run on a Mac Mini M4:

| Config | Simulations |
|---|---|
| `--batches 5 --rounds 100` | ~8,000 |
| `--batches 10 --rounds 200` | ~32,000 |
| `--batches 20 --rounds 1000 --workers 8` | ~320,000 |
| Overnight unattended (8–12 hrs) | 500K–1M+ |

`--workers N` runs rounds in parallel across CPU cores. Stage 1 costs nothing — no API calls, runs as fast as your CPU.

---

## Three Stages

**Stage 1 — Evolutionary (free)**
16 candidate strategies per batch. Top winners mutate, cross over, and compete again. Categorical specialists are preserved so the engine doesn't collapse to one average policy. No API calls. Runs as fast as your CPU.

**Stage 2 — AI-directed (~$0.50/run)**
Opus 4.6 with adaptive thinking directs between batches — diagnoses what the simulator is teaching vs. gaming, retires weak principles, and focuses the next batch on known gaps. Sonnet generates 16 named strategy archetypes per batch. Haiku extracts conditional principles every 10 rounds. Every 2 batches, the director generates adversarial scenarios targeting the champion's weak spots and injects them into the next batch.

**Stage 3 — Local model (free forever)**
Export the tournament log as training data. Numerical domains: XGBoost on `training_features.csv`. Language domains: fine-tune Qwen3-4B on Apple Silicon via MLX-LM. `python run.py train` handles both. No more API calls.

```
bootstrap → Stage 1 → Stage 2 → saturated → export → local specialist
  ~$0.05      free      ~$0.50     free          forever
```

---

## Full Run

```bash
# Stage 1 — free, no API key needed
python run.py run --domain FreightQuoting --batches 10 --rounds 150

# Stage 2 — AI archetypes + director (~$0.50)
export ANTHROPIC_API_KEY=sk-ant-...
python run.py run --domain FreightQuoting --brain --batches 8 --rounds 150

# Fully autonomous overnight run — promotes from Stage 1 to Stage 2 automatically
python run.py run --domain FreightQuoting --auto --batches 20 --rounds 1000 --workers 8
```

Run overnight unattended:

```bash
caffeinate -i python run.py run \
  --domain FreightQuoting \
  --auto --batches 20 --rounds 1000 --workers 8 \
  > FreightQuoting/run.log 2>&1 &

echo "Running as PID $! — check FreightQuoting/run.log in the morning"
```

```bash
# Morning check
python run.py tail   --domain FreightQuoting
python run.py status --domain FreightQuoting
python run.py eval   --domain FreightQuoting
python run.py export --domain FreightQuoting
python run.py train  --domain FreightQuoting
```

Checkpointed after every batch — safe to interrupt and resume. Delete `run_checkpoint.json` to start fresh.

---

## Convergence

The director issues a verdict after every batch:

| Verdict | Meaning | Action |
|---|---|---|
| `converging` | Score and playbook improving | Keep running |
| `exploring` | Mixed results, still searching | Keep running |
| `stalled` | No improvement for multiple batches | Adjust sim or prompts |
| `reward_hacking` | Score rising but wrong behavior winning | Stop — fix simulation.py |
| `needs_calibration` | Sim rewarding the wrong thing | Fix simulation.py |
| `saturated` | Playbook full, score stable | Export and train |

`saturated` is the success state. When `reward_hacking` or `needs_calibration` fires, the director prints specific fix suggestions for `simulation.py` and logs them in `thinking_log.md`.

---

## Abstention — When Not to Act

Every round logs a `score_margin` (winner minus runner-up). Low margin means even the simulation wasn't confident. After export, `training_features.csv` includes a `score_margin` column and an `uncertain` flag:

```python
ABSTAIN_THRESHOLD = 0.05  # printed by export — bottom 25% of training margins
scores = model.predict(xgb.DMatrix(candidate_rows))
if scores.max() - np.median(scores) < ABSTAIN_THRESHOLD:
    return {"action": "ABSTAIN", "reason": "strategies too close — escalate to human"}
best = candidates[int(np.argmax(scores))]
```

A specialist that knows when to escalate ships faster than one that always answers.

---

## Stage 3: Local Model

**Numerical domain** (prices, rates, scores, demand signals):
```bash
python run.py train --domain FreightQuoting
# Trains XGBoost on training_features.csv, saves model.json
```

**Language domain** (text classification, routing, document scoring):
```bash
python run.py train --domain TicketTriage
# Prints the exact mlx_lm.lora command for your domain

mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \
             --data TicketTriage/ --train --iters 1000

mlx_lm.fuse --model mlx-community/Qwen3-4B-4bit \
             --adapter-path TicketTriage/adapters --save-path TicketTriage/fused-model

ollama create ticket-triage -f - <<EOF
FROM ./TicketTriage/fused-model
EOF
ollama run ticket-triage "scenario: ..."
```

```bash
# Install training dependencies when you're ready for Stage 3
pip install -e ".[train-numeric]"   # XGBoost — runs on any hardware
pip install -e ".[train-language]"  # MLX-LM — requires Apple Silicon
```

Stage 3 XGBoost runs on any hardware. MLX-LM fine-tuning requires Apple Silicon.

---

## Domain Packs — Share a Vertical

Domains are versioned, shareable packages:

```bash
python run.py pack FreightQuoting
# → FreightQuoting-1.0.0.zip

python run.py install FreightQuoting-1.0.0.zip
```

Verify before using:

```bash
python run.py eval --domain FreightQuoting
# ✓ spot_high_demand     score  6.2  min  2.0
# ✓ spot_low_margin      score  4.1  min  1.8
# ✗ contract_volatile    score  1.4  min  2.6  ← weak spot
# 2/3 passed
```

`eval` exits with code 1 on failure — use it as a CI gate.

---

## Production Flywheel

Once deployed, real decisions close the loop:

```bash
# Import real outcomes (format: {"state": {...}, "decision": {...}, "outcome": 4.2})
python run.py import --domain FreightQuoting --file live_trades.jsonl

python run.py export --domain FreightQuoting
python run.py train  --domain FreightQuoting
```

simulate → train → deploy → import real outcomes → retrain → repeat

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
    ├── pack.json               version, author, metric                                ← commit this
    ├── evals/scenarios.jsonl   known scenarios + expected scores                      ← commit this
    ├── playbook.jsonl          learned conditional principles                         ← commit this
    ├── top_candidates.json     top Stage 1 winners                                    ← commit this
    ├── training_preferences.jsonl  pairwise winner-vs-runner-up examples              (generated)
    ├── thinking_log.md         director reasoning trail                               (do not commit)
    ├── tournament_log.jsonl    round-by-round scores                                  (do not commit)
    └── last_run.json           last run summary                                       (do not commit)
```

The Autoforge root has zero domain knowledge. `simulation.py` is the only file that knows what your domain is.

---

## Model Configuration

Built on Anthropic. Opus 4.6 with adaptive thinking for the director, Sonnet 4.6 for archetypes, Haiku 4.5 for extraction. Stage 2 calls the API directly via stdlib `urllib` — no SDK required.

```bash
ANTHROPIC_API_KEY=sk-ant-...
AUTOFORGE_AI_BACKEND=anthropic          # or: manual

# Override any model (optional — defaults shown)
AUTOFORGE_DIRECTOR_MODEL=claude-opus-4-6
AUTOFORGE_LIBRARY_MODEL=claude-sonnet-4-6
AUTOFORGE_EXTRACT_MODEL=claude-haiku-4-5-20251001
```

`AUTOFORGE_AI_BACKEND=manual` runs without live API calls — useful for testing.

OpenAI backend is available but prompts are tuned for Claude; adaptive thinking is not supported. See [issue #1](https://github.com/eyoellundberg/autoforge/issues/1).

---

## Cost

| Stage | Cost | Notes |
|---|---|---|
| `bootstrap` | ~$0.05 | One-time per domain |
| Stage 1 | $0.00 | No API calls |
| Stage 2 | ~$0.50 | Per run to saturation |
| Stage 3 | $0.00 | Local compute, forever |

Designed to run unattended on a Mac Mini M4. No GPU needed for Stage 1 or 2.

---

## License

MIT — see [LICENSE](LICENSE).
