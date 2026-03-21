# Autoforge

**Describe a decision. Get a deployed specialist. No labeled data.**

One command turns a plain-English description into a fine-tuned local model — freight quoting, fraud detection, grain marketing, anything with a scorable outcome. Autoforge writes the simulator, runs competing strategies, distills what it learned into training data, and fine-tunes a Qwen3-4B specialist you drop into your app.

## Quick start

```bash
pip install autoforge
pip install mlx mlx-lm      # Apple Silicon — for local fine-tuning (once)
```

```bash
$ autoforge

  What do you want to call it?  > GrainExpert
  What does it do?              > grain marketing decisions for midwest corn farmers

  Planning GrainExpert...

  Metric:     expected_profit
  Parameters: basis_threshold, carry_weight, hedge_ratio, storage_capacity... (24 total)
  Scenarios:  SA crop status, seasonal timing, cash flow pressure, basis level...
  Tension:    wide carry favors holding, but cash flow and basis risk favor selling

  Generate GrainExpert? [Y/n]

Bootstrap   ✔  world model written                                      (~$0.30)
Validate    ✔  simulation looks good
Train       ████░░░░  batch 3/8...                                      (~$0.50)
Fine-tune   ████░░░░  Qwen3-4B LoRA on Apple Silicon...                 (free, ~20 min)

Done. Specialist lives at ~/.autoforge/GrainExpert/specialist/
```

AutoForge disappears after this. GrainExpert is yours.

## How it works

```
describe → simulate → compete → extract → fine-tune → deploy
```

**Bootstrap.** Opus acts as a domain expert first — identifying the 20-35 variables that actually drive decisions in your domain, then writing a generative simulator that captures all of them. Not a toy with 5 parameters. A real world model from day one.

**Tournament.** Strategies compete across hundreds of thousands of random scenarios. What consistently wins becomes your training signal. The simulation is the teacher. You never label a single example.

**Extract.** Conditional principles are distilled from each batch — *"when carry is wide and South America is at risk, hold rather than sell"* — and a director steers the next batch toward unexplored territory. Breakthrough discoveries (findings that reshape how other principles should be interpreted) are flagged and weighted more heavily in training.

**Fine-tune.** Tournament results are verbalized into natural language training examples and used to fine-tune Qwen3-4B locally via LoRA. The specialist learns to reason about your domain, not just look up a score.

**Deploy.** The specialist ships as a standalone folder at `~/.autoforge/GrainExpert/specialist/`. No Autoforge, no API calls, no framework. The engine stays in `~/.autoforge/GrainExpert/` if you want to keep training.

## Python SDK

If you already have a scoring function, skip the CLI entirely:

```python
from autoforge import run

champion = run(simulate=my_simulate, state=my_state, schema=SCHEMA)

print(champion.strategy)    # {"basis_threshold": -0.35, "carry_weight": 0.7, ...}
print(champion.philosophy)  # "Hold when carry is positive and SA risk is elevated"
print(champion.playbook)    # [{"principle": "...", "confidence": 0.87}, ...]
```

## The specialist

Training produces a standalone module — no Autoforge dependency, no API calls.

Query it from the terminal:
```bash
autoforge ask --domain GrainExpert "basis is -0.35, SA drought risk, October"
# → Hold. SA drought risk in October overrides the slightly negative basis...
```

Or use it in code:
```python
from specialist.ask import ask, record

result = ask({"basis": -0.35, "carry": 0.02, "south_america": "drought_risk"})
# → "Hold. SA drought risk overrides the slightly negative basis.
#    Holding 4-6 weeks improves basis by $0.06-0.11 in ~75% of cases..."

record(features, actual_outcome)   # log real outcomes for retraining
```

Build whatever you want on top — API, iOS app, Slack bot, agent tool. The specialist has no idea AutoForge exists.

Retrains automatically on real outcomes:
```
0 2 * * * cd /your/app && python specialist/retrain.py
```

Autoforge is scaffolding. You remove it when the building stands.

## What changed in v2

The engine was simplified significantly. Everything that was compensating for weak foundations got removed:

- **Self-evolve** — gone. Opus now builds a rich, expert-level simulation at bootstrap. No need to patch it mid-run.
- **Reality grounding** — gone. Same reason.
- **Hypothesis tracking** — gone. The playbook already captures what was learned.
- **Batch promotion logic** — gone. Batch 1 is procedural exploration, batch 2+ uses AI archetypes. Automatic, no flags.
- **Year management** — gone. Was a relic from a specific domain, never belonged in the engine.
- **Adversarial injection** — gone. A rich simulation covers the space naturally.
- **XGBoost specialist** — gone. Replaced with a fine-tuned Qwen3-4B that reasons in natural language, gives numbers, and explains its decisions.

What was added:

- **Breakthrough detection** — the director classifies findings that reshape the domain model vs. incremental improvements. Breakthrough batches get 2x weight in fine-tuning.
- **Verbalized training data** — tournament results are turned into natural language examples (scenario + recommended strategy + reasoning chain + expected outcome) before fine-tuning.

The loop is now: generate → score → extract → direct → repeat. ~600 lines of engine. Domains build complexity on top if they need it.

## Cost

| | Cost |
|---|---|
| Bootstrap (one-time) | ~$0.30 |
| Training run | ~$0.50 |
| Fine-tune Qwen3-4B (local) | free, ~20 min on Apple Silicon |
| Specialist inference + retraining | free forever |

Runs unattended on a Mac Mini. No GPU needed.

## License

MIT
