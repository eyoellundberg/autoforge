# Autoforge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Claude](https://img.shields.io/badge/powered%20by-Claude-orange.svg)](https://anthropic.com)

**Describe a decision. Get a deployed specialist. No labeled data.**

One command turns a plain-English description into a trained local model that makes domain-specific decisions — freight quoting, fraud detection, loan approval, anything with a scorable outcome. Autoforge writes the simulator, runs hundreds of thousands of competing strategies, and ships a standalone specialist you deploy and retrain on real outcomes. The AI builds the expertise, then gets out of the way.

---

## Quickstart

```bash
uv pip install -e .         # or: pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...

autoforge FreightQuoting "freight brokerage load quoting, spot and contract lanes"
```

That's it. Autoforge will:

```
● [1/3] Bootstrap domain           — AI writes the simulator         (~$0.30)
● [2/3] Validate simulation        — checks schema, score range       (free)
● [3/3] Train                      — Stage 1 → Stage 2 → specialist  (~$0.50)

✔ Done in 12m 34s.

  Deploy:   cp -r FreightQuoting/specialist/ /your/app/
  Use:      from specialist.predict import predict, record
  Retrain:  python retrain.py
```

When it finishes, `FreightQuoting/specialist/` contains 4 files. Copy them anywhere. Zero Autoforge dependency.

---

## The Specialist

Training produces a standalone module — no framework, no API calls, no Autoforge needed:

```
specialist/
  predict.py     # predict() + record()
  retrain.py     # retrains on real outcomes
  model.json     # trained weights
  config.json    # features, params, threshold
```

Two functions:

```python
from specialist.predict import predict, record

# Make a decision
result = predict({"lane_distance": 450, "weight": 42000, "volatility": 0.12})
# → {"strategy": {"bid_margin": 0.12, ...}, "score": 6.4}
# → {"action": "ABSTAIN", "reason": "strategies too close — escalate"}

# Record what actually happened
record(scenario, result["strategy"], actual_profit)
```

The specialist knows when to abstain. Low-confidence scenarios get escalated instead of answered.

### Retraining

The specialist retrains itself on real outcomes. No Autoforge, no API calls:

```bash
python retrain.py
# Merges original training data + real outcomes → new model.json
```

Schedule it as a cron job:

```bash
# Every Sunday — model gets smarter from its own decisions
0 3 * * 0 cd /your/app/specialist && python retrain.py
```

Autoforge is scaffolding. You remove it when the building stands. Come back only when the domain shifts fundamentally and you need to re-explore.

---

## How It Works

```
describe → simulate → compete → extract → deploy → retrain on reality
```

1. **You describe the domain** — "freight brokerage load quoting"
2. **AI writes a generative simulator** — `simulate()` returns expected value: `P(outcome | features) x magnitude`
3. **16 strategies compete** across hundreds of thousands of scenarios
4. **Conditional principles extracted** — "when volatility > 20% and lane is new, conservative margins win"
5. **A hypothesis-driven director** tracks what the engine is learning, retires weak principles, and steers exploration
6. **XGBoost trained on winners** — exported as a standalone specialist with abstention

The simulation is the teacher. Strategies compete against it. What consistently wins becomes your training data. You never label a single example.

---

## What You Can Train It On

Anything where you can score a decision against a scenario.

**Decision-making under uncertainty:**
- Quote a freight load at the right margin
- Enter a stock position given technical signals
- Accept or reject a loan application
- Bid on a programmatic ad impression
- Triage a patient for urgency level

**Optimization with a clear objective:**
- Pricing strategy across customer segments
- Inventory reorder timing
- Resource allocation across queues

**Policy learning where data is expensive or slow:**
- Clinical decision support (simulate patient outcomes)
- Financial crime compliance (simulate detection accuracy)
- Supply chain response to disruption scenarios

---

## Three Stages

**Stage 1 — Evolutionary (free)**
16 candidate strategies per batch. Top winners mutate, cross over, and compete. Categorical specialists are preserved so the engine doesn't collapse to one average policy. No API calls.

**Stage 2 — AI-directed (~$0.50/run)**
Opus directs between batches — tests hypotheses about the domain, diagnoses what the simulator is teaching vs. gaming, retires weak principles, focuses the next batch on known gaps. Sonnet generates 16 named strategy archetypes. Haiku extracts conditional principles every 10 rounds. Every 2 batches, adversarial scenarios target the champion's weak spots.

**Stage 3 — Deployed specialist (free forever)**
The specialist runs locally. XGBoost for numerical domains, fine-tuned Qwen3-4B via MLX-LM for language domains. Retrains on real outcomes without Autoforge.

```
describe → Stage 1 → Stage 2 → saturated → specialist/
 ~$0.30      free      ~$0.50     auto        forever
```

---

## The World Model

`world_model.md` is the single steering document for the entire learning loop. Generated by bootstrap, it contains:

- **Domain Understanding** — how the domain works in reality, key dynamics, failure modes
- **Strategy Space** — what dimensions matter, what range of approaches to explore
- **Extraction Guidance** — what principles are worth learning vs. simulation artifacts
- **Success Criteria** — what good looks like, when to abstain, what must never happen
- **Current Hypotheses** — what the engine is testing, what it has confirmed

Every AI call in the system — the director, the archetype generator, the principle extractor, the grounding layer — reads this document. One coherent understanding drives everything.

---

## Hypothesis-Driven Learning

The director doesn't just optimize — it tests hypotheses about the domain.

After each batch, it tracks:
- **hypotheses_tested** — what was tested this batch
- **hypotheses_confirmed** — what the evidence supports
- **hypotheses_open** — what to test next

Confirmed hypotheses accumulate in `hypotheses.json` and update the Current Hypotheses section of `world_model.md`. The engine builds a theory of the domain, not just a set of weights.

---

## Self-Evolution

With `--self-evolve`, the engine can improve its own simulation:

```bash
autoforge FreightQuoting "freight load quoting" --self-evolve
```

When the director detects that `simulate()` is miscalibrated — rewarding the wrong behavior, missing real-world dynamics, or has dead parameters — it proposes a patch. Opus rewrites `simulation.py`, validates it passes all checks, and the tournament continues with the improved sim.

Schema evolution works the same way: the director can add or remove parameters from `CANDIDATE_SCHEMA` when it discovers new strategic dimensions or finds dead ones.

---

## Reality Grounding

With `--ground`, the engine sanity-checks the simulation against domain knowledge:

```bash
autoforge FreightQuoting "freight load quoting" --ground
```

Three layers:
1. **Anomaly detection** (pure stats, every batch) — score outliers, dead parameters, uniform scores, batch drift
2. **Calibration audit** (Sonnet, every 3rd batch) — "are these probability distributions realistic for this domain?"
3. **Deep-dive sampling** (Haiku, every batch) — random rounds reviewed for real-world plausibility

If the calibration verdict is "miscalibrated" and `--self-evolve` is on, the engine automatically rewrites the simulation.

---

## The Simulator Contract

`simulation.py` is the only file that knows what your domain is:

```python
CANDIDATE_SCHEMA: dict         # JSON schema for a strategy
METRIC_NAME: str               # e.g. "expected_profit"

def random_state() -> dict:    # draw one scenario
def simulate(candidate: dict, state: dict) -> float:  # score it

# Optional
def build_context(state: dict) -> dict:  # labels for principle extraction
```

The best simulations are generative world models — `simulate()` returns expected value: `P(good_outcome | features) x magnitude`. When the simulation models the real-world outcome, the tournament discovers strategies that work for grounded reasons.

---

## CLI

**Primary interface** — one command does everything:

```bash
autoforge Domain "description"           # create + train → specialist/
autoforge Domain                         # status, or resume if interrupted
autoforge Domain --eval                  # run eval scenarios
autoforge Domain --pack                  # bundle as .zip
autoforge Domain --import outcomes.jsonl # import real outcomes
autoforge install pack.zip               # install a domain pack
```

**Power-user subcommands** for granular control:

```bash
autoforge run --domain D --brain --batches 8 --rounds 200
autoforge run --domain D --brain --self-evolve --ground
autoforge calibrate --domain D
autoforge validate --domain D
autoforge export --domain D
autoforge train --domain D
autoforge tail --domain D
autoforge status --domain D
autoforge bootstrap D "description" --manual
```

**Overnight unattended:**

```bash
caffeinate -i autoforge FreightQuoting \
  "freight brokerage load quoting" \
  --batches 20 --rounds 1000 --workers 8 \
  > FreightQuoting/run.log 2>&1 &
```

Checkpointed after every batch — safe to interrupt and resume.

---

## Scale

| Config | Simulations |
|---|---|
| `--batches 5 --rounds 100` | ~8,000 |
| `--batches 10 --rounds 200` | ~32,000 |
| `--batches 20 --rounds 1000 --workers 8` | ~320,000 |
| Overnight unattended (8–12 hrs) | 500K–1M+ |

`--workers N` runs rounds in parallel. Stage 1 costs nothing.

---

## Convergence

The director issues a verdict after every batch:

| Verdict | Meaning | Action |
|---|---|---|
| `converging` | Score and playbook improving | Keep running |
| `exploring` | Mixed results, still searching | Keep running |
| `stalled` | No improvement for multiple batches | Adjust sim |
| `reward_hacking` | Score rising but wrong behavior winning | Fix simulation.py |
| `needs_calibration` | Sim rewarding the wrong thing | Fix simulation.py |
| `saturated` | Playbook full, score stable | Done — specialist generated |

`saturated` is the success state. On saturation, the engine automatically exports training data, trains XGBoost, and generates `specialist/`.

---

## Production Lifecycle

```
Day 1        autoforge FreightQuoting "freight load quoting"
             → specialist/ deployed to production

Week 2       specialist retrains on real outcomes (python retrain.py)
Week 3       specialist retrains again — getting smarter
Month 2      still retraining weekly, no Autoforge needed

Month 6      domain shifted — come back to Autoforge
             autoforge FreightQuoting --import outcomes.jsonl
             autoforge run --domain FreightQuoting --brain --batches 3
             autoforge train --domain FreightQuoting
             → specialist v2 deployed
```

Autoforge gets you from zero to a working model when you have no data. Once you have real outcomes, the specialist retrains itself. Autoforge comes back when you need to re-explore.

---

## Domain Packs

Share trained domains:

```bash
autoforge FreightQuoting --pack
# → FreightQuoting-1.0.0.zip

autoforge install FreightQuoting-1.0.0.zip
autoforge FreightQuoting --eval
```

---

## Architecture

Flat — one file per concern, no subdirectories:

```
autoforge/
├── cli.py          CLI entry point + subcommand dispatch
├── run.py          tournament orchestration + checkpoint/resume
├── bootstrap.py    domain generation from description
├── validate.py     simulation health: validate, calibrate, generate evals
├── tools.py        domain management: status, tail, pack, install, eval, import, train
├── progress.py     pipeline progress display
├── tournament.py   core batch runner (Stage 1 + Stage 2)
├── brain.py        archetypes (Sonnet) + adversarial + principle extraction (Haiku)
├── export.py       training data exporter
├── evolve.py       simulation + schema self-evolution
├── ground.py       reality grounding (anomaly detection, calibration audit)
├── director.py     hypothesis-driven director (Opus)
├── api.py          AI calls via stdlib urllib — no SDK
├── utils.py        file helpers, ENGINE_ROOT, git
└── template/
    ├── simulation.py       domain template
    ├── world_model.md      steering document template
    └── specialist/         predict.py + retrain.py templates

Domain/
├── simulation.py           the generative world model
├── world_model.md          unified steering document
├── hypotheses.json         confirmed + open hypotheses
├── playbook.jsonl          learned conditional principles
├── specialist/             deployable: predict.py, retrain.py, model.json, config.json
├── pack.json               version, metric, metadata
└── evals/scenarios.jsonl   eval scenarios
```

---

## Model Configuration

Built on Anthropic. No SDK required — calls the API via stdlib `urllib`.

```bash
ANTHROPIC_API_KEY=sk-ant-...

# Override any model (optional — defaults shown)
AUTOFORGE_BOOTSTRAP_MODEL=claude-opus-4-6        # domain generation
AUTOFORGE_DIRECTOR_MODEL=claude-opus-4-6         # hypothesis-driven direction
AUTOFORGE_LIBRARY_MODEL=claude-sonnet-4-6        # archetype generation
AUTOFORGE_EXTRACT_MODEL=claude-haiku-4-5-20251001 # principle extraction
AUTOFORGE_EVOLVE_MODEL=claude-opus-4-6           # simulation self-evolution
AUTOFORGE_GROUND_MODEL=claude-sonnet-4-6         # calibration audit
AUTOFORGE_DEEP_DIVE_MODEL=claude-haiku-4-5-20251001 # deep-dive sampling
```

---

## Cost

| Stage | Cost | Notes |
|---|---|---|
| Bootstrap | ~$0.30 | One-time (Opus) |
| Stage 1 | $0.00 | No API calls |
| Stage 2 | ~$0.50 | Per run to saturation |
| Specialist | $0.00 | Local, forever |
| Retraining | $0.00 | XGBoost on real outcomes |

Runs unattended on a Mac Mini M4. No GPU needed.

---

## License

MIT — see [LICENSE](LICENSE).
