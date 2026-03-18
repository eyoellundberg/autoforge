"""
commands/bootstrap.py — cmd_bootstrap and cmd_new subcommands.
"""

import json
import shutil
import sys

from commands.shared import ENGINE_ROOT, BOOTSTRAP_SCHEMA


def cmd_bootstrap(args):
    """Generate a new domain from a natural language description using Sonnet."""
    template_path = ENGINE_ROOT / "template"
    domain_path   = ENGINE_ROOT / args.domain

    if domain_path.exists():
        print(f"Error: {domain_path} already exists.")
        sys.exit(1)
    if not template_path.exists():
        print("Error: template/ folder not found.")
        sys.exit(1)

    print(f"\nBootstrapping {args.domain} from description...")
    print("Calling Sonnet to generate domain files...\n")

    system_prompt = """You are designing a simulation domain for Autoforge, an autonomous strategy learning system.

Autoforge works by:
1. Generating a library of 16 named strategy archetypes (via Sonnet)
2. Running a deterministic simulation that scores each archetype against random scenarios
3. Extracting conditional principles from what wins
4. A director AI reading results and improving the next archetype library

You will generate all files needed to define a new domain. Each file must be complete and runnable.

SIMULATION.PY REQUIREMENTS:
- simulate(candidate: dict, state: dict) -> float   # deterministic, fast, no I/O
- random_state() -> dict                             # draws one scenario from the domain distribution
- CANDIDATE_SCHEMA: dict                             # JSON schema describing a strategy's parameters
- METRIC_NAME: str                                   # e.g. "profit", "score", "accuracy"

CALIBRATION RULES:
- simulate() must be deterministic — same inputs always produce same output
- random_state() must cover diverse scenarios — different scenario types should favor different strategies
- The score range should be reasonable (e.g. 0-200, not 1e9)
- No single strategy should dominate all scenarios — variety is essential for learning
- Include an is_event or equivalent flag for unusual/extreme scenarios

CANDIDATE_SCHEMA RULES:
- Must be a valid JSON schema object
- Parameters should be numeric ranges or small enums
- Include 3-6 meaningful dimensions that a strategy can vary
- Each dimension should meaningfully affect the score in simulate()

PROMPT RULES:
- brain.md: instructions for generating 16 diverse strategy archetypes. Name each archetype distinctively. Cover the full strategy space including contrarian approaches.
- extract.md: instructions for extracting 0-2 conditional principles per round. Include {{RETIRED_TOPICS}} placeholder.
- director.md: context about the domain for the between-batch director. What to watch for, what failure modes exist.
"""

    user_prompt = f"""Generate all domain files for Autoforge based on this description:

{args.description}

Requirements:
- simulation_py: complete, runnable Python file with all required exports
- brain_md: complete prompt for generating 16 strategy archetypes for this domain
- extract_md: complete prompt for principle extraction (include {{{{RETIRED_TOPICS}}}} placeholder)
- director_md: domain context for the batch director
- context_keys: list of key names from random_state() that should appear in _build_context()
- calibration_notes: specific things to check/tune in the simulation before running the full tournament
- metric_name: the metric being optimized
- domain_summary: 1-2 sentence description of what this domain learns
"""

    import anthropic
    client = anthropic.Anthropic()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": BOOTSTRAP_SCHEMA,
                }
            },
        )
        data = json.loads(response.content[0].text)
    except Exception as e:
        print(f"Error: Anthropic API call failed: {e}")
        # Clean up partial domain folder if it was created
        if domain_path.exists():
            shutil.rmtree(domain_path)
        sys.exit(1)

    # 1. Copy template/ to domain_path/
    shutil.copytree(template_path, domain_path)

    try:
        # 2. Write the generated simulation.py (overwrite template skeleton)
        (domain_path / "simulation.py").write_text(data["simulation_py"])

        # 3. Write brain_md, extract_md, director_md to prompts/
        prompts_path = domain_path / "prompts"
        prompts_path.mkdir(exist_ok=True)
        (prompts_path / "brain.md").write_text(data["brain_md"])
        (prompts_path / "extract.md").write_text(data["extract_md"])
        (prompts_path / "director.md").write_text(data["director_md"])

        # 4. Generate _build_context() body from context_keys — edit tournament.py
        context_keys = data["context_keys"]
        tournament_path = domain_path / "tournament.py"
        tournament_text = tournament_path.read_text()

        # Build the new _build_context() body lines
        return_lines = []
        for key in context_keys:
            return_lines.append(f'        "{key}": state.get("{key}"),')
        return_body = "\n".join(return_lines)

        old_build_context = '''def _build_context(state: dict) -> dict:
    """
    Build the context dict passed to the AI extractor.

    These key-value pairs describe the scenario in plain terms — the AI reads
    them to extract conditional principles ("IF demand is high AND competition is low...").

    Make these human-readable. The AI's extraction quality depends on this.
    """
    # TODO: adapt this to your actual state dict structure
    return {
        "demand":      state.get("demand", "unknown"),
        "competition": state.get("competition", "unknown"),
        "context_flag": state.get("context_flag", False),
        # Add more domain-relevant context here
    }'''

        new_build_context = f'''def _build_context(state: dict) -> dict:
    """
    Build the context dict passed to the AI extractor.

    These key-value pairs describe the scenario in plain terms — the AI reads
    them to extract conditional principles ("IF demand is high AND competition is low...").

    Make these human-readable. The AI's extraction quality depends on this.
    """
    return {{
{return_body}
    }}'''

        tournament_text = tournament_text.replace(old_build_context, new_build_context)
        tournament_path.write_text(tournament_text)

        # 5. Write .env with ANTHROPIC_API_KEY placeholder
        (domain_path / ".env").write_text("ANTHROPIC_API_KEY=sk-ant-...\n")

        # 6. Create data/ directory inside domain_path
        (domain_path / "data").mkdir(exist_ok=True)
        (domain_path / "data" / ".gitkeep").touch()

    except Exception as e:
        print(f"Error: Failed to write domain files: {e}")
        shutil.rmtree(domain_path)
        sys.exit(1)

    domain_summary    = data["domain_summary"]
    calibration_notes = data["calibration_notes"]

    print(f"Created {args.domain}/\n")
    print(domain_summary)
    print(f"\nCalibration notes:\n{calibration_notes}")
    print(f"""
IMPORTANT: Review {args.domain}/simulation.py before running.
The simulation is the only thing the engine learns from — get it right.

Next:
  python run.py run --domain {args.domain} --batches 3 --rounds 50   # sanity check
  python run.py run --domain {args.domain} --brain --batches 5 --rounds 150""")


def cmd_new(args):
    """Scaffold a new domain from the template."""
    template_path = ENGINE_ROOT / "template"
    if not template_path.exists():
        print(f"Error: template/ folder not found at {template_path}")
        sys.exit(1)

    domain_path = ENGINE_ROOT / args.domain
    if domain_path.exists():
        print(f"Error: {domain_path} already exists.")
        sys.exit(1)

    shutil.copytree(template_path, domain_path)

    env_file = domain_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-...\n")

    print(f"\nCreated {args.domain}/")
    print()
    print("Next steps:")
    print(f"  1. Edit {args.domain}/simulation.py — implement simulate(), random_state(), CANDIDATE_SCHEMA, METRIC_NAME")
    print(f"  2. Edit {args.domain}/prompts/brain.md — tell Sonnet what archetypes to generate")
    print(f"  3. Edit {args.domain}/prompts/extract.md — tell Haiku what principles to extract")
    print(f"  4. Edit {args.domain}/prompts/director.md — context for the batch director")
    print(f"  5. Set your API key in {args.domain}/.env")
    print()
    print("Then run:")
    print(f"  python run.py run --domain {args.domain}                  # Stage 1, no API calls")
    print(f"  python run.py run --domain {args.domain} --brain          # Stage 2, AI archetypes")
    print(f"  python run.py run --domain {args.domain} --auto           # Stage 1 → Stage 2 auto")
