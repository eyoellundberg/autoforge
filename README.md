# Playbook ML

**Describe a decision. Get a deployed specialist. No labeled data.**

One command turns a plain-English description into a fine-tuned local model — freight quoting, fraud detection, grain marketing, anything with a scorable outcome. Playbook ML writes the simulator, runs competing strategies, distills what it learned into training data, and fine-tunes a Qwen3-4B specialist you drop into your app.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install playbook-ml
```

## Run

```bash
playbook-ml
```

```
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

Done. Specialist lives at ~/.playbook-ml/GrainExpert/specialist/
```

Playbook ML disappears after this. GrainExpert is yours.

## How it works

```
describe → simulate → compete → extract → fine-tune → deploy
```

**Bootstrap.** Opus acts as a domain expert first — identifying the 20-35 variables that actually drive decisions in your domain, then writing a generative simulator that captures all of them. Not a toy with 5 parameters. A real world model from day one.

**Tournament.** Strategies compete across hundreds of thousands of random scenarios. What consistently wins becomes your training signal. The simulation is the teacher. You never label a single example.

**Extract.** Conditional principles are distilled from each batch — *"when carry is wide and South America is at risk, hold rather than sell"* — and a director steers the next batch toward unexplored territory. Breakthrough discoveries (findings that reshape how other principles should be interpreted) are flagged and weighted more heavily in training.

**Fine-tune.** Tournament results are verbalized into natural language training examples and used to fine-tune Qwen3-4B locally via LoRA. The specialist learns to reason about your domain, not just look up a score.

**Deploy.** The specialist ships as a standalone folder at `~/.playbook-ml/GrainExpert/specialist/`. No Playbook ML, no API calls, no framework. The engine stays in `~/.playbook-ml/GrainExpert/` if you want to keep training.

## The specialist

Training produces a standalone module — no Playbook ML dependency, no API calls.

Query it from the terminal:
```bash
playbook-ml ask --domain GrainExpert "basis is -0.35, SA drought risk, October"
# → Hold. SA drought risk in October overrides the slightly negative basis...
```

Or drop it into your app:
```python
from specialist.ask import ask, record

result = ask({"basis": -0.35, "carry": 0.02, "south_america": "drought_risk"})
# → "Hold. SA drought risk overrides the slightly negative basis.
#    Holding 4-6 weeks improves basis by $0.06-0.11 in ~75% of cases..."

record(features, actual_outcome)   # log real outcomes for retraining
```

Build whatever you want on top — API, iOS app, Slack bot, agent tool. The specialist has no idea Playbook ML exists.

Retrains automatically on real outcomes:
```
0 2 * * * cd /your/app && python specialist/retrain.py
```

Playbook ML is scaffolding. You remove it when the building stands.

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
