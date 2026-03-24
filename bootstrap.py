"""
bootstrap.py — cmd_bootstrap: generate a new domain from a description.
"""

import itertools
import json
import os
import shutil
import sys
import threading
import time

from utils import ENGINE_ROOT, DOMAINS_ROOT
from director import BOOTSTRAP_SCHEMA
from api import structured_ai_call
from validate import validate_domain, _print_validation


PREVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "metric":      {"type": "string",
                        "description": "The single metric simulate() returns, e.g. 'profit', 'routing_accuracy'"},
        "parameters":  {"type": "array", "items": {"type": "string"},
                        "description": "5-12 candidate strategy parameter names — the levers a strategy can tune"},
        "scenarios":   {"type": "array", "items": {"type": "string"},
                        "description": "5-10 key factors that vary across scenarios in random_state()"},
        "scores":      {"type": "string",
                        "description": "One sentence: what does simulate() actually compute and return?"},
        "tension":     {"type": "string",
                        "description": "One sentence: what is the core tension that prevents any single strategy from always winning?"},
    },
    "required": ["metric", "parameters", "scenarios", "scores", "tension"],
    "additionalProperties": False,
}


def cmd_bootstrap(args):
    """Generate a new domain from a description, or scaffold manually with --manual."""
    template_path = ENGINE_ROOT / "template"
    domain_path   = DOMAINS_ROOT / args.domain

    if domain_path.exists():
        print(f"Error: {domain_path} already exists.")
        sys.exit(1)
    if not template_path.exists():
        print("Error: template/ folder not found.")
        sys.exit(1)

    # --manual: just copy the template, no AI
    if getattr(args, "manual", False):
        shutil.copytree(template_path, domain_path)
        (domain_path / ".env").write_text("# ANTHROPIC_API_KEY=sk-ant-...\n")
        (domain_path / "data").mkdir(exist_ok=True)
        (domain_path / "data" / ".gitkeep").touch()
        print(f"\nCreated {args.domain}/")
        print(f"\nNext:")
        print(f"  1. Edit {args.domain}/world_model.md")
        print(f"  2. Edit {args.domain}/simulation.py")
        print(f"  3. playbook-ml run --domain {args.domain}")
        return

    if not args.description:
        print("Error: description is required (or use --manual for template-only scaffold).")
        sys.exit(1)

    if getattr(args, "manual_ai", False):
        os.environ["PLAYBOOK_ML_AI_BACKEND"] = "manual"

    # ── Preview: cheap Haiku call to outline the domain design ────────────────
    if not getattr(args, "yes", False):
        print(f"\nPlanning {args.domain}...")
        preview_system = (
            "You are planning a simulation domain for an autonomous strategy learning system. "
            "Be concrete and specific to the domain described."
        )
        preview_user = (
            f"Domain description: {args.description}\n\n"
            "Outline the simulation design: what metric it optimises, "
            "what parameters a strategy can tune, what scenario factors vary, "
            "what simulate() computes, and what tension prevents one strategy from always winning."
        )
        try:
            preview = structured_ai_call(
                task_name="preview",
                domain_path=ENGINE_ROOT,
                model=os.environ.get("PLAYBOOK_ML_EXTRACT_MODEL", "claude-haiku-4-5-20251001"),
                max_tokens=512,
                system_prompt=preview_system,
                user_prompt=preview_user,
                schema=PREVIEW_SCHEMA,
            )
            print(f"\n  Metric:     {preview['metric']}")
            print(f"  Parameters: {', '.join(preview['parameters'])}")
            print(f"  Scenarios:  {', '.join(preview['scenarios'])}")
            print(f"  Scores:     {preview['scores']}")
            print(f"  Tension:    {preview['tension']}")
        except Exception:
            print("  (preview unavailable — proceeding to full generation)")

        print()
        try:
            answer = input("Generate full domain with Opus? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)
        if answer not in ("", "y", "yes"):
            print("Aborted.")
            sys.exit(0)

    print(f"\nGenerating {args.domain}...")

    system_prompt = """You are a domain expert AND simulation designer for Playbook ML, an autonomous strategy learning system.

Your job: think deeply about this domain as an expert practitioner, then build a simulation rich enough
that a learning engine can discover real strategy principles from it.

STEP 1 — THINK LIKE AN EXPERT:
Before writing any code, ask yourself:
- What are ALL the variables that actually drive decisions in this domain?
- What do real practitioners worry about that novices miss?
- What are the non-obvious interactions? (e.g. variable A only matters when B is high)
- What makes this domain hard? What prevents any single strategy from always winning?

CANDIDATE_SCHEMA — THE STRATEGY SPACE:
Identify 20-35 variables that expert practitioners actually use when making decisions.
Not generic placeholders — the real levers. For grain marketing: basis, carry, cash flow,
South America crop status, seasonal timing, hedge ratio, storage capacity. All of them.
Each must meaningfully affect the outcome in simulate().

SIMULATION.PY REQUIREMENTS:
- simulate(candidate: dict, state: dict) -> float   # deterministic, fast, no I/O
- random_state() -> dict                             # draws one rich scenario from the domain distribution
- CANDIDATE_SCHEMA: dict                             # JSON schema — all the strategy levers
- METRIC_NAME: str                                   # e.g. "expected_profit", "accuracy", "roi"

SIMULATION DESIGN PRINCIPLE:
The simulation is a generative world model, not a scoring function.
simulate() returns expected value: P(good_outcome | strategy, state) × magnitude.

Examples:
- Grain marketing: P(profitable_sale | basis, carry, timing) × expected_margin
- Freight pricing: P(winning_bid | price, market_rate) × margin_if_won
- Loan approval: P(repayment) × revenue - P(default) × loss
- Fraud detection: P(fraud | signals) × prevented_loss - P(false_positive) × friction_cost

CALIBRATION RULES:
- simulate() must be deterministic — same inputs always produce same output
- random_state() must generate rich, diverse scenarios that favor different strategies
- Score range should be reasonable (not 1e9)
- No single strategy should dominate all scenarios — tension is essential for learning
- Use empirically-calibrated distributions reflecting how this domain actually behaves

WORLD MODEL — agent instructions for brain, director, extractor, and specialist.
Write this as a precise specification, not a human explainer. Agents act on it directly.

## Variables
List every state variable with: type, range, key thresholds, and what it signals.
Format:
  name: description. Range: X to Y. Threshold: Z (what crossing Z means).
Example:
  carry: cost of storing grain per month. Range: -0.02 to +0.08.
         Threshold: +0.02 (above = storage pays, hold bias increases sharply).

## Decision Rules
Explicit IF/THEN conditionals that govern good decisions in this domain.
These are the patterns the specialist must learn. Include interactions.
Format:
  IF [condition] → [action] because [mechanism]
Example:
  IF carry > 0.02 AND sa_risk < elevated → hold bias (storage paying, supply risk ahead)
  IF basis < -0.40 → sell regardless of carry (basis collapse overrides all)

## Strategy Space (brain instructions)
Directives for generating 16 archetypes. Name the archetype types that MUST exist.
What dimensions span the space? What is the contrarian bet? What combos are interesting?
Be explicit: "generate one archetype that ignores carry entirely and trades only on basis momentum"

## Extraction Guidance (extractor instructions)
Explicit patterns worth extracting vs artifacts to reject.
Format:
  EXTRACT: IF [context] AND [condition] → [outcome] (conditional, domain-specific, has mechanism)
  REJECT:  any principle claiming X always beats Y — sim cannot verify unconditional dominance
  WATCH:   [context factor] × [context factor] interactions — these produce the most valuable principles
Include the placeholder: {{RETIRED_TOPICS}}

## Director Watchlist
Specific failure modes for this domain the director must flag:
  - [exact condition that indicates reward hacking in this sim]
  - [exact condition that indicates mode collapse]
  - [exact condition that indicates a sim artifact principle]

## Success Criteria
Job: [one sentence — what decision does this specialist make]
Excellent: [specific, numeric — what good outcomes look like]
Never: [hard constraints — what must never happen]
Abstain when: [conditions where the specialist should escalate to a human]
"""

    user_prompt = f"""Generate all domain files for Playbook ML based on this description:

{args.description}

Requirements:
- simulation_py: complete, runnable Python file with all required exports. Should be a generative
  world model returning expected value, not an arbitrary scoring function.
- world_model_md: unified steering document with all sections (Domain Understanding, Strategy Space,
  Extraction Guidance, Success Criteria). Include {{{{RETIRED_TOPICS}}}} placeholder
  in the Extraction Guidance section.
- context_keys: list of key names from random_state() that should appear in build_context()
- calibration_notes: specific things to check/tune in the simulation before running
- metric_name: the metric being optimized (should reflect expected value, e.g. "expected_profit")
- domain_summary: 1-2 sentence description of what this domain learns
"""

    def _spinner(label: str, stop: threading.Event):
        frames = itertools.cycle(["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"])
        start  = time.time()
        while not stop.is_set():
            elapsed = int(time.time() - start)
            sys.stdout.write(f"\r  {next(frames)}  {label}  {elapsed}s")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    try:
        backend_domain_path = DOMAINS_ROOT
        _stop = threading.Event()
        _t    = threading.Thread(
            target=_spinner,
            args=("Opus is thinking — building world model...", _stop),
            daemon=True,
        )
        _t.start()
        try:
            data = structured_ai_call(
                task_name="bootstrap",
                domain_path=backend_domain_path,
                model=os.environ.get("PLAYBOOK_ML_BOOTSTRAP_MODEL", "claude-opus-4-6"),
                thinking=True,
                max_tokens=16000,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=BOOTSTRAP_SCHEMA,
                metadata={"domain": args.domain},
            )
        finally:
            _stop.set()
            _t.join()
    except Exception as e:
        print(f"Error: bootstrap AI call failed: {e}")
        if domain_path.exists():
            shutil.rmtree(domain_path)
        sys.exit(1)

    # 1. Copy template/ to domain_path/
    shutil.copytree(template_path, domain_path)

    try:
        # 2. Write the generated simulation.py (overwrite template skeleton)
        (domain_path / "simulation.py").write_text(data["simulation_py"])

        # 3. Write world_model.md — the unified steering document
        (domain_path / "world_model.md").write_text(data["world_model_md"])

        # 4. Append build_context() to simulation.py using context_keys
        context_keys = data["context_keys"]
        context_lines = "\n".join(
            f'        "{k}": state.get("{k}"),' for k in context_keys
        )
        build_context_fn = f'''\n\ndef build_context(state: dict) -> dict:
    """
    Human-readable scenario description for AI principle extraction.
    These key-value pairs appear in the extractor prompt as conditional context.
    """
    return {{
{context_lines}
    }}\n'''
        sim_text = (domain_path / "simulation.py").read_text()
        (domain_path / "simulation.py").write_text(sim_text + build_context_fn)

        # 5. Write .env with ANTHROPIC_API_KEY placeholder
        (domain_path / ".env").write_text("# ANTHROPIC_API_KEY=sk-ant-...\n")

        # 6. Create data/ directory inside domain_path
        (domain_path / "data").mkdir(exist_ok=True)
        (domain_path / "data" / ".gitkeep").touch()

        # 7. Write pack.json manifest
        pack = {
            "name": args.domain,
            "version": "1.0.0",
            "author": "",
            "description": data["domain_summary"],
            "metric": data["metric_name"],
            "playbook_ml_version": "1.0",
            "evals": "evals/scenarios.jsonl",
        }
        (domain_path / "pack.json").write_text(json.dumps(pack, indent=2) + "\n")

        # 8. Create evals/ folder with empty scenarios file
        (domain_path / "evals").mkdir(exist_ok=True)
        (domain_path / "evals" / "scenarios.jsonl").write_text(
            "# Add eval scenarios here. Format: {\"id\": \"...\", \"state\": {...}, \"description\": \"...\", \"min_score\": 0}\n"
        )

        # 9. Auto-generate eval scenarios from world model
        try:
            from validate import generate_evals
            evals = generate_evals(domain_path, n=8)
            if evals:
                print(f"\n  Generated {len(evals)} eval scenarios")
        except Exception:
            pass  # eval generation is best-effort

    except Exception as e:
        print(f"Error: Failed to write domain files: {e}")
        shutil.rmtree(domain_path)
        sys.exit(1)

    domain_summary    = data["domain_summary"]
    calibration_notes = data["calibration_notes"]

    print(f"Created {args.domain}/\n")
    print(domain_summary)
    print(f"\nCalibration notes:\n{calibration_notes}")

    # Auto-validate the generated simulation
    print(f"\nValidating {args.domain}/simulation.py...")
    ok, warnings, errors = validate_domain(domain_path)
    _print_validation(ok, warnings, errors)

    if errors:
        print(f"\n! Simulation has errors — review {args.domain}/simulation.py before running.")
        print(f"  playbook-ml validate --domain {args.domain}   # re-check after edits")
    else:
        print(f"""
Next:
  1. Review {args.domain}/world_model.md  — the steering document for the learning loop
  2. Review {args.domain}/simulation.py   — check the variables look right for your domain

  playbook-ml calibrate --domain {args.domain}        # check score range + no single winner
  playbook-ml {args.domain}                           # run the full training pipeline
""")
