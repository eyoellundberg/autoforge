"""
bootstrap.py — cmd_bootstrap: generate a new domain from a description.
"""

import json
import os
import shutil
import sys

from utils import ENGINE_ROOT, load_world_model
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
    domain_path   = ENGINE_ROOT / args.domain

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
        print(f"  3. autoforge run --domain {args.domain}")
        return

    if not args.description:
        print("Error: description is required (or use --manual for template-only scaffold).")
        sys.exit(1)

    if getattr(args, "manual_ai", False):
        os.environ["AUTOFORGE_AI_BACKEND"] = "manual"

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
                model=os.environ.get("AUTOFORGE_EXTRACT_MODEL", "claude-haiku-4-5-20251001"),
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

    system_prompt = """You are designing a simulation domain for Autoforge, an autonomous strategy learning system.

Autoforge works by:
1. Building a world model — a generative model of how the domain works in reality
2. Generating strategy archetypes that explore different approaches (via Sonnet)
3. Running a deterministic simulation that scores each strategy against random scenarios
4. Extracting conditional principles from what wins (via Haiku)
5. A director AI tracking hypotheses, analyzing results, and directing exploration
6. Optionally evolving the simulation itself when gaps are discovered

You will generate simulation.py and a unified world_model.md steering document.

SIMULATION.PY REQUIREMENTS:
- simulate(candidate: dict, state: dict) -> float   # deterministic, fast, no I/O
- random_state() -> dict                             # draws one scenario from the domain distribution
- CANDIDATE_SCHEMA: dict                             # JSON schema describing a strategy's parameters
- METRIC_NAME: str                                   # e.g. "expected_value", "profit", "accuracy"

SIMULATION DESIGN PRINCIPLE:
The best simulations are generative world models, not scoring functions.
simulate() should return expected value: P(good_outcome | features) × magnitude.
When simulate() models the real-world outcome, the tournament discovers strategies
that work for grounded reasons, not arbitrary assumptions.

Examples:
- Fraud detection: P(fraud | signals) × cost_of_fraud - P(false_positive) × cost_of_flag
- Pitch scoring: P(meaningful_outcome | team, traction, market) × expected_return
- Freight bidding: P(winning_bid | price, market) × margin
- Loan approval: P(repayment) × interest_revenue - P(default) × loss_given_default

CALIBRATION RULES:
- simulate() must be deterministic — same inputs always produce same output
- random_state() must cover diverse scenarios — different types should favor different strategies
- The score range should be reasonable (not 1e9)
- No single strategy should dominate all scenarios — variety is essential for learning
- Use empirically-calibrated probability distributions where possible

CANDIDATE_SCHEMA RULES:
- Must be a valid JSON schema object
- Parameters should be numeric ranges or small enums
- Include 3-6 meaningful dimensions that a strategy can vary
- Each dimension should meaningfully affect the score in simulate()

WORLD MODEL RULES:
world_model_md must be a single steering document with these sections:

## Domain Understanding
What is this domain? How does it work in reality? What are the key dynamics,
tensions, and failure modes? What should the AI director watch for during training?

## Strategy Space
How should strategy archetypes be generated? What dimensions matter?
What range of approaches should be explored? Include instructions for generating
16 diverse archetypes covering the full strategy space.

## Extraction Guidance
What principles are worth learning? What context factors matter?
What patterns are real domain knowledge vs simulation artifacts?
Include the placeholder: {{RETIRED_TOPICS}}

## Success Criteria
What decision does the specialist make? What does good look like?
When should it abstain? What must never happen?

## Current Hypotheses
Start with: "- (none yet — populated after first batch)"
Then end the section with these exact lines (the engine writes hypothesis updates between them):
<!-- hypotheses-start -->
<!-- hypotheses-end -->
"""

    user_prompt = f"""Generate all domain files for Autoforge based on this description:

{args.description}

Requirements:
- simulation_py: complete, runnable Python file with all required exports. Should be a generative
  world model returning expected value, not an arbitrary scoring function.
- world_model_md: unified steering document with all sections (Domain Understanding, Strategy Space,
  Extraction Guidance, Success Criteria, Current Hypotheses). Include {{{{RETIRED_TOPICS}}}} placeholder
  in the Extraction Guidance section.
- context_keys: list of key names from random_state() that should appear in build_context()
- calibration_notes: specific things to check/tune in the simulation before running
- metric_name: the metric being optimized (should reflect expected value, e.g. "expected_profit")
- domain_summary: 1-2 sentence description of what this domain learns
"""

    try:
        backend_domain_path = domain_path.parent if domain_path.parent.exists() else ENGINE_ROOT
        data = structured_ai_call(
            task_name="bootstrap",
            domain_path=backend_domain_path,
            model=os.environ.get("AUTOFORGE_BOOTSTRAP_MODEL", "claude-opus-4-6"),
            thinking=True,
            max_tokens=8000,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=BOOTSTRAP_SCHEMA,
            metadata={"domain": args.domain},
        )
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
            "autoforge_version": "1.0",
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
        print(f"  autoforge validate --domain {args.domain}   # re-check after edits")
    else:
        print(f"""
Next:
  1. Review {args.domain}/world_model.md  — the steering document for the entire learning loop
  2. Review {args.domain}/simulation.py   — the generative world model the engine learns from

  autoforge calibrate --domain {args.domain}               # check score range + dominance
  autoforge run       --domain {args.domain} --batches 5 --rounds 100
  autoforge run       --domain {args.domain} --brain --batches 5 --rounds 150
  autoforge run       --domain {args.domain} --brain --self-evolve --batches 8 --rounds 200""")
