"""
commands/cli.py — Autoforge CLI argument parsing and dispatch.

Five commands that cover the full lifecycle: scaffold a domain, run the
tournament, inspect state, export training data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bootstrap <Domain> --description "..."
    Generate a complete domain from a natural language description.
    Sonnet writes simulation.py, all three prompt files, and adapts
    tournament.py's _build_context() automatically. One-time cost ~$0.05.
    Always review simulation.py before running — it is the only thing
    the engine ever learns from.

    python run.py bootstrap GrainMarketing \\
      --description "corn/soy marketing, midwest US farms"

new <Domain>
    Scaffold a new domain by copying template/ verbatim.
    Use this when you want to write simulation.py yourself from scratch.
    Produces an empty skeleton with comments explaining each required export.

    python run.py new GrainMarketing

calibrate --domain <Domain> [--n N]
    Sample N random scenarios, show distributions for each state key, score
    8 random candidates, and flag calibration issues (dominant strategies,
    zero-heavy scores, low variance). Run this after editing simulation.py
    before committing to a long run.

    python run.py calibrate --domain GrainMarketing
    python run.py calibrate --domain GrainMarketing --n 1000

run --domain <Domain> [options]
    Run the tournament. Loads tournament.py from the domain folder and
    calls run_batch() for each batch. Between batches, the director AI
    (Sonnet) reads results, appends to thinking_log.md, and sets hints
    for the next batch's archetype generation.

    Modes:
      (default)   Stage 1 — evolutionary mutation, no API calls.
                  _generate_procedural_candidates() evolves from prior batch
                  winners using elitism, mutation, crossover, and random fill.
                  Playbook grows from Haiku extraction every 10 rounds.

      --brain     Stage 2 — Sonnet generates 16 named strategy archetypes
                  per batch. Each has a philosophy, not just parameters.
                  Haiku extracts principles every 10 rounds. Director reads
                  results and retires losers between batches. Champion
                  archetype propagates to seed the next batch. ~$0.50/run.

      --auto      Stage 1 until playbook flatlines for 2 consecutive batches,
                  then auto-promotes to Stage 2. Fully autonomous overnight run.

    Options:
      --batches N   batches to run per year (default: 8)
      --rounds N    rounds per batch (default: 200)
      --years N     years to run — playbook carries across years (default: 1)
      --workers N   parallel workers for simulation scoring (default: 1)
                    use os.cpu_count() to run all cores in parallel

    Examples:
      python run.py run --domain GrainMarketing --batches 10 --rounds 150
      python run.py run --domain GrainMarketing --brain --batches 8 --rounds 150
      python run.py run --domain GrainMarketing --auto --batches 20 --rounds 150
      python run.py run --domain GrainMarketing --brain --years 2 --batches 5
      python run.py run --domain GrainMarketing --workers 4 --batches 8 --rounds 200

    Stops early if the director returns verdict "saturated" (success) or
    "reward_hacking" (stop and fix the sim). Writes last_run.json on exit.

export --domain <Domain>
    Export Stage 3 training data from tournament_log.jsonl.
    Quality filter: keeps rounds scoring >= median across all rounds.
    Output: training_data.jsonl — messages-format JSONL (system/user/assistant).

      system:    domain context + playbook principles
      user:      scenario state dict
      assistant: winning strategy JSON

    Compatible with MLX-LM (fine-tune Qwen on Apple Silicon) and
    OpenAI-compatible fine-tuning endpoints.

    python run.py export --domain GrainMarketing

status --domain <Domain>
    Show current domain state in a Rich terminal table:
      - Top 5 playbook principles by confidence
      - Champion archetype name and philosophy
      - Retired topics (permanent blocklist)
      - Last run timestamp, round count, final verdict
      - tournament_log.jsonl size + training_data.jsonl status

    python run.py status --domain GrainMarketing

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTOR VERDICTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  converging        score and playbook improving — keep running
  exploring         mixed results, still searching — keep running
  stalled           no improvement — adjust sim or prompts
  reward_hacking    score rising for wrong reason — stop, fix the sim
  needs_calibration sim rewarding wrong behavior — fix simulation.py
  saturated         playbook full, score stable — export and train

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CONFIGURATION (optional, set in MyDomain/.env)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ENGINE_DIRECTOR_MODEL   between-batch director  (default: claude-sonnet-4-6)
  ENGINE_LIBRARY_MODEL    archetype generation    (default: claude-sonnet-4-6)
  ENGINE_EXTRACT_MODEL    principle extraction    (default: claude-haiku-4-5-20251001)

  Swap Sonnet for Haiku everywhere to cut cost ~10x.
  Use Opus for the director for more rigorous analysis.
"""

import argparse

from commands.bootstrap import cmd_bootstrap, cmd_new
from commands.run_cmd import cmd_run
from commands.tools import cmd_calibrate, cmd_validate, cmd_export, cmd_status


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Autoforge — autonomous strategy learning system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_bootstrap = subparsers.add_parser("bootstrap", help="Generate a new domain from a description using AI")
    p_bootstrap.add_argument("domain", help="Domain name (e.g. GrainMarketing)")
    p_bootstrap.add_argument("--description", required=True, help="Natural language description of the domain")

    # new
    p_new = subparsers.add_parser("new", help="Scaffold a new domain from template")
    p_new.add_argument("domain", help="Domain name (e.g. GrainMarketing)")

    # run
    p_run = subparsers.add_parser("run", help="Run the tournament")
    p_run.add_argument("--domain",  required=True, help="Domain subfolder name")
    p_run.add_argument("--batches", type=int, default=8,   help="Batches per year (default 8)")
    p_run.add_argument("--rounds",  type=int, default=200, help="Rounds per batch (default 200)")
    p_run.add_argument("--years",   type=int, default=1,   help="Years to run (default 1)")
    p_run.add_argument("--brain",   action="store_true",   help="Stage 2: AI archetypes")
    p_run.add_argument("--auto",    action="store_true",   help="Stage 1 until saturated, then auto-promote to Stage 2")
    p_run.add_argument("--workers", type=int, default=1,
        help="Parallel workers for simulation (default 1, use os.cpu_count() for max)")

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Show scenario distributions and score stats")
    p_cal.add_argument("--domain", required=True, help="Domain subfolder name")
    p_cal.add_argument("--n", type=int, default=500, help="Scenarios to sample (default 500)")

    # validate
    p_validate = subparsers.add_parser("validate", help="Sanity-check a domain's simulation.py")
    p_validate.add_argument("--domain", required=True, help="Domain subfolder name")

    # export
    p_export = subparsers.add_parser("export", help="Export Stage 3 training data")
    p_export.add_argument("--domain", required=True, help="Domain subfolder name")

    # status
    p_status = subparsers.add_parser("status", help="Show domain state")
    p_status.add_argument("--domain", required=True, help="Domain subfolder name")

    args = parser.parse_args()

    if args.command == "bootstrap":
        cmd_bootstrap(args)
    elif args.command == "new":
        cmd_new(args)
    elif args.command == "calibrate":
        cmd_calibrate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "status":
        cmd_status(args)
