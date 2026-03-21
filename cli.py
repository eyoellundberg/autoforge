"""
cli.py — Autoforge CLI entry point.

Primary interface:
  autoforge Domain "description"       # create + train (full pipeline)
  autoforge Domain                     # status or resume
  autoforge Domain --eval              # run eval scenarios
  autoforge Domain --pack              # bundle as .zip
  autoforge Domain --import file.jsonl # import real outcomes
  autoforge install pack.zip           # install a domain pack

Power-user subcommands (granular control):
  autoforge bootstrap, run, calibrate, validate, export, train, status, tail

Environment overrides:
  AUTOFORGE_DIRECTOR_MODEL
  AUTOFORGE_LIBRARY_MODEL
  AUTOFORGE_EXTRACT_MODEL
  AUTOFORGE_EVALS_MODEL
  AUTOFORGE_BOOTSTRAP_MODEL
"""

import argparse
import sys

from utils import ENGINE_ROOT, DOMAINS_ROOT
from bootstrap import cmd_bootstrap
from run import cmd_run
from validate import cmd_calibrate, cmd_validate
from tools import (
    cmd_export, cmd_status, cmd_pack, cmd_install, cmd_eval,
    cmd_import, cmd_tail, cmd_train, cmd_generate_evals, cmd_ask,
)


_SUBCOMMANDS = {
    "bootstrap", "run", "calibrate", "validate", "export", "status",
    "pack", "install", "eval", "import", "tail", "train", "generate-evals", "ask",
}


def _cmd_go(args):
    """The unified pipeline: bootstrap (if needed) → validate → run → train."""
    from progress import Pipeline

    domain_path = DOMAINS_ROOT / args.domain

    if not domain_path.exists():
        if not args.description:
            print(f"Domain '{args.domain}' not found.")
            print(f"\nCreate it:")
            print(f"  autoforge {args.domain} \"describe your domain here\"")
            print(f"  autoforge {args.domain} --manual")
            sys.exit(1)

    if not args.description and domain_path.exists():
        if getattr(args, "eval", False):
            cmd_eval(args)
        elif getattr(args, "pack", False):
            cmd_pack(args)
        elif getattr(args, "import_file", None):
            args.file = args.import_file
            cmd_import(args)
        elif (domain_path / "run_checkpoint.json").exists():
            print(f"Run in progress — resuming...\n")
            import os
            args.batches = getattr(args, "batches", 30)
            args.rounds = getattr(args, "rounds", 200)
            args.manual_ai = False
            if not getattr(args, "workers", None) or args.workers <= 1:
                args.workers = min(max(1, (os.cpu_count() or 2) - 1), 8)
            cmd_run(args)
        else:
            cmd_status(args)
        return

    if domain_path.exists() and args.description is None:
        return

    # ── Full pipeline with progress display ──────────────────────────────

    needs_bootstrap = not domain_path.exists()
    stages = []
    if needs_bootstrap:
        stages.append("Bootstrap domain")
    stages += [
        "Validate simulation",
        "Train — tournament → specialist",
    ]

    pipe = Pipeline(args.domain, stages)

    if needs_bootstrap:
        with pipe.stage():
            cmd_bootstrap(args)

    with pipe.stage():
        from validate import validate_domain, _print_validation
        ok, warnings, errors = validate_domain(domain_path)
        _print_validation(ok, warnings, errors)
        if errors:
            print(f"\n  Fix errors in {args.domain}/simulation.py before running.")
            sys.exit(1)

    with pipe.stage():
        import os
        args.batches = getattr(args, "batches", 30)
        args.rounds = getattr(args, "rounds", 200)
        args.manual_ai = getattr(args, "manual_ai", False)
        if not getattr(args, "workers", None) or args.workers <= 1:
            args.workers = min(max(1, (os.cpu_count() or 2) - 1), 8)
        cmd_run(args)

    pipe.summary()

    spec_dir = domain_path / "specialist"
    if spec_dir.exists():
        print(f"\n  Deploy:   cp -r {args.domain}/specialist/ /your/app/")
        print(f"  Use:      from specialist.ask import ask, record")
        print(f"  Retrain:  python retrain.py  (on real outcomes, no Autoforge needed)")
    print()


def _cmd_interactive():
    """No args: prompt for name and description, then run the full pipeline."""
    import os
    print("\nAutoforge\n")
    try:
        name = input("  What do you want to call it?  > ").strip()
        if not name:
            print("Name required.")
            sys.exit(1)
        description = input("  What does it do?           > ").strip()
        if not description:
            print("Description required.")
            sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    args = argparse.Namespace(
        domain=name,
        description=description,
        yes=False,
        manual=False,
        batches=8,
        rounds=200,
        workers=min(max(1, (os.cpu_count() or 2) - 1), 8),
        eval=False,
        pack=False,
        import_file=None,
    )
    _cmd_go(args)


def main():
    # No args → interactive setup
    if len(sys.argv) == 1:
        return _cmd_interactive()

    # Check if first arg is a known subcommand → power-user mode
    if len(sys.argv) > 1 and sys.argv[1] in _SUBCOMMANDS:
        return _main_subcommands()

    # Primary interface: autoforge Domain ["description"] [flags]
    parser = argparse.ArgumentParser(
        prog="autoforge",
        description="Autoforge — autonomous strategy learning system",
        usage="autoforge Domain [\"description\"] [options]",
    )
    parser.add_argument("domain", help="Domain name (e.g. FreightQuoting)")
    parser.add_argument("description", nargs="?", default=None,
                        help="Natural language description — creates + trains the domain")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--manual", action="store_true", help="Scaffold without AI")
    parser.add_argument("--batches", type=int, default=8, help="Batches (default 8)")
    parser.add_argument("--rounds", type=int, default=200, help="Rounds per batch (default 200)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1)")
    parser.add_argument("--eval", action="store_true", help="Run eval scenarios")
    parser.add_argument("--pack", action="store_true", help="Bundle as .zip")
    parser.add_argument("--import", dest="import_file", metavar="FILE", help="Import real outcomes")

    args = parser.parse_args()
    _cmd_go(args)


def _main_subcommands():
    """Power-user mode: explicit subcommands for granular control."""
    parser = argparse.ArgumentParser(
        prog="autoforge",
        description="Autoforge — power-user subcommands",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_bootstrap = subparsers.add_parser("bootstrap", help="Generate domain from description")
    p_bootstrap.add_argument("domain", help="Domain name")
    p_bootstrap.add_argument("description", nargs="?", default=None, help="Domain description")
    p_bootstrap.add_argument("--yes", "-y", action="store_true")
    p_bootstrap.add_argument("--manual", action="store_true", help="Template only, no AI")
    p_bootstrap.add_argument("--manual-ai", action="store_true")

    # run
    p_run = subparsers.add_parser("run", help="Run tournament")
    p_run.add_argument("--domain", required=True)
    p_run.add_argument("--batches", type=int, default=8)
    p_run.add_argument("--rounds", type=int, default=200)
    p_run.add_argument("--workers", type=int, default=1)

    # calibrate
    p_cal = subparsers.add_parser("calibrate", help="Score distribution stats")
    p_cal.add_argument("--domain", required=True)
    p_cal.add_argument("--n", type=int, default=500)

    # validate
    p_val = subparsers.add_parser("validate", help="Sanity-check simulation.py")
    p_val.add_argument("--domain", required=True)

    # export
    p_exp = subparsers.add_parser("export", help="Export training data")
    p_exp.add_argument("--domain", required=True)

    # status
    p_st = subparsers.add_parser("status", help="Show domain state")
    p_st.add_argument("--domain", required=True)

    # pack
    p_pack = subparsers.add_parser("pack", help="Bundle domain as .zip")
    p_pack.add_argument("domain")

    # install
    p_inst = subparsers.add_parser("install", help="Install domain pack")
    p_inst.add_argument("pack", help="Path to .zip")
    p_inst.add_argument("--force", action="store_true")

    # eval
    p_eval = subparsers.add_parser("eval", help="Run eval scenarios")
    p_eval.add_argument("--domain", required=True)

    # import
    p_imp = subparsers.add_parser("import", help="Import real outcomes")
    p_imp.add_argument("--domain", required=True)
    p_imp.add_argument("--file", required=True)

    # tail
    p_tail = subparsers.add_parser("tail", help="Live snapshot of in-progress run")
    p_tail.add_argument("--domain", required=True)

    # train
    p_train = subparsers.add_parser("train", help="Train Stage 3 model")
    p_train.add_argument("--domain", required=True)

    # generate-evals
    p_ge = subparsers.add_parser("generate-evals", help="Auto-generate eval scenarios")
    p_ge.add_argument("--domain", required=True)
    p_ge.add_argument("--n", type=int, default=10)

    # ask
    p_ask = subparsers.add_parser("ask", help="Query the specialist from the terminal")
    p_ask.add_argument("--domain", required=True)
    p_ask.add_argument("question", help="Question or feature dict (JSON string)")

    args = parser.parse_args()

    dispatch = {
        "bootstrap":      cmd_bootstrap,
        "run":            cmd_run,
        "calibrate":      cmd_calibrate,
        "validate":       cmd_validate,
        "export":         cmd_export,
        "status":         cmd_status,
        "pack":           cmd_pack,
        "install":        cmd_install,
        "eval":           cmd_eval,
        "import":         cmd_import,
        "tail":           cmd_tail,
        "train":          cmd_train,
        "generate-evals": cmd_generate_evals,
        "ask":            cmd_ask,
    }
    dispatch[args.command](args)
