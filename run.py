"""
run.py — Tournament orchestration loop.

autoforge run --domain Domain --batches 8 --rounds 200
"""

import json
import sys

from utils import ENGINE_ROOT, DOMAINS_ROOT, load_env, retire_principles, git_commit_batch, append_breakthroughs
from director import call_director


def _default_analysis(hints=None):
    return {
        "verdict": "exploring",
        "hints": hints or [],
        "retire_principles": [],
        "next_batch_focus": "",
        "observations": [],
        "principles_gaining_confidence": [],
        "concerns": [],
        "mistakes_to_note": [],
        "simulation_fix_suggestions": [],
        "breakthroughs": [],
    }


def cmd_run(args):
    domain_path = DOMAINS_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    load_env(domain_path)
    from tournament import run_batch

    # Resume from checkpoint if present
    checkpoint_path = domain_path / "run_checkpoint.json"
    generation_offset = 0
    hints             = []
    prior_analysis    = None
    playbook_sizes    = []
    start_batch       = 1
    use_brain         = False

    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            if ckpt.get("domain") == args.domain:
                generation_offset = ckpt["generation_offset"]
                hints             = ckpt.get("hints", [])
                prior_analysis    = ckpt.get("prior_analysis")
                playbook_sizes    = ckpt.get("playbook_sizes", [])
                start_batch       = ckpt.get("batch_num", 0) + 1
                use_brain         = ckpt.get("use_brain", False)
                print(f"Resuming from batch {start_batch - 1} (delete run_checkpoint.json to start fresh)\n")
        except Exception:
            pass

    worker_str = f"{args.workers} workers" if args.workers > 1 else "1 worker"
    print(f"\nAutoforge — {args.domain}")
    print(f"  {args.rounds} rounds/batch  ·  {args.batches} batches  ·  {worker_str}")
    print()

    for batch_num in range(start_batch, args.batches + 1):
        # First batch: procedural exploration. After that: AI archetypes.
        if not use_brain and batch_num > 1:
            use_brain = True

        print(f"  {batch_num:>2}  running...", end="\r", flush=True)

        try:
            result = run_batch(
                domain_path,
                args.rounds,
                generation_offset=generation_offset,
                hints=hints,
                use_brain=use_brain,
                workers=args.workers,
                batch_num=batch_num,
            )
        except Exception as e:
            print(f"\n  [batch {batch_num} error: {e}] — skipping")
            generation_offset += args.rounds
            continue

        generation_offset += args.rounds
        playbook_sizes.append(result["playbook_size"])

        try:
            analysis = call_director(batch_num, result, prior_analysis, domain_path, playbook_sizes)
            verdict  = analysis["verdict"]
        except Exception:
            analysis = _default_analysis(hints=hints)
            verdict  = "exploring"

        # Batch summary line
        prev_avg  = (prior_analysis or {}).get("_avg_score")
        delta_str = f"  Δ{result['avg_score'] - prev_avg:+.2f}" if prev_avg is not None else ""
        print(f"  {batch_num:>2}  avg {result['avg_score']}  {result['trend_pct']:+.1f}%{delta_str}  [{verdict}]")

        focus = (analysis.get("next_batch_focus") or "")[:80]
        if focus and verdict not in ("converging", "saturated"):
            print(f"      → {focus}")
        if verdict in ("reward_hacking", "needs_calibration") and analysis.get("simulation_fix_suggestions"):
            for fix in analysis["simulation_fix_suggestions"][:3]:
                print(f"      ! {fix}")
        for bt in analysis.get("breakthroughs", []):
            print(f"      ★ breakthrough: {bt['principle'][:100]}")

        hints          = analysis.get("hints", [])
        prior_analysis = {**analysis, "_avg_score": result["avg_score"]}
        retire_principles(analysis, domain_path)
        append_breakthroughs(domain_path, batch_num, analysis.get("breakthroughs", []))
        git_commit_batch(args.domain, domain_path, batch_num, result, analysis)

        checkpoint_path.write_text(json.dumps({
            "domain":            args.domain,
            "generation_offset": generation_offset,
            "batch_num":         batch_num,
            "use_brain":         use_brain,
            "hints":             hints,
            "prior_analysis":    prior_analysis,
            "playbook_sizes":    playbook_sizes,
        }, indent=2))

        if verdict == "saturated":
            print("\nPlaybook saturated — engine has learned everything this sim can teach.")
            break
        if verdict == "reward_hacking":
            print(f"\n! Reward hacking detected — fix {args.domain}/simulation.py and re-run calibrate.")
            break

    checkpoint_path.unlink(missing_ok=True)
    print("\a", end="", flush=True)

    pb_path = domain_path / "playbook.jsonl"
    n_principles = sum(1 for _ in pb_path.open()) if pb_path.exists() else 0
    champion_name = ""
    champ_path = domain_path / "champion_archetype.json"
    if champ_path.exists():
        try:
            champion_name = json.loads(champ_path.read_text()).get("name", "")
        except Exception:
            pass

    verdict = (prior_analysis or {}).get("verdict", "unknown")
    print(f"\n  {'─'*46}")
    print(f"  {args.domain}  ·  {verdict}  ·  {generation_offset:,} rounds")
    if n_principles:
        print(f"  {n_principles} principles", end="")
        if champion_name:
            print(f"  ·  champion: {champion_name}", end="")
        print()
    print(f"  {'─'*46}")


if __name__ == "__main__":
    from cli import main
    main()
