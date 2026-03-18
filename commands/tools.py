"""
commands/tools.py — cmd_calibrate, cmd_validate, cmd_status, cmd_export subcommands.
"""

import json
import sys

from rich.table import Table

from commands.shared import ENGINE_ROOT, console


def _print_validation(ok: list, warnings: list, errors: list):
    for msg in ok:
        print(f"  [ok] {msg}")
    for msg in warnings:
        print(f"  [warn] {msg}")
    for msg in errors:
        print(f"  [error] {msg}")


def cmd_calibrate(args):
    """Show scenario distributions, score stats, and dominance check."""
    import importlib
    import os
    import statistics
    import collections

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)

    try:
        sim = importlib.import_module("simulation")
    except Exception as e:
        print(f"simulation.py import failed: {e}")
        sys.exit(1)

    n = args.n
    print(f"\nCalibrating {args.domain} — sampling {n} scenarios...\n")

    states = [sim.random_state() for _ in range(n)]

    import random as _random
    schema_props = sim.CANDIDATE_SCHEMA.get("properties", {})

    def random_candidate():
        c = {}
        for key, spec in schema_props.items():
            if spec.get("type") == "number":
                lo, hi = spec.get("minimum", 0.0), spec.get("maximum", 1.0)
                c[key] = _random.uniform(lo, hi)
            elif spec.get("type") == "integer":
                lo, hi = spec.get("minimum", 0), spec.get("maximum", 10)
                c[key] = _random.randint(lo, hi)
            elif "enum" in spec:
                c[key] = _random.choice(spec["enum"])
            else:
                c[key] = None
        return c

    candidates = [random_candidate() for _ in range(8)]
    win_counts = collections.Counter()
    all_scores = []

    for state in states:
        scored = [(sim.simulate(c, state), i) for i, c in enumerate(candidates)]
        scored.sort(reverse=True)
        all_scores.append(scored[0][0])
        win_counts[scored[0][1]] += 1

    # ── Scenario distribution ─────────────────────────────────────────────────
    tbl = Table(title=f"Scenario Distribution ({n} samples)", show_header=True)
    tbl.add_column("Key")
    tbl.add_column("Type")
    tbl.add_column("Distribution")

    for k in list(states[0].keys()):
        vals = [s[k] for s in states if k in s]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            true_pct = sum(1 for v in vals if v) / len(vals) * 100
            tbl.add_row(k, "bool", f"{true_pct:.0f}% True  /  {100-true_pct:.0f}% False")
        elif isinstance(vals[0], (int, float)):
            tbl.add_row(k, "number",
                f"min {min(vals):.2f}  median {statistics.median(vals):.2f}  max {max(vals):.2f}")
        else:
            counts = collections.Counter(vals)
            top = sorted(counts.items(), key=lambda x: -x[1])[:6]
            tbl.add_row(k, "categorical",
                "  ".join(f"{v}: {c/len(vals)*100:.0f}%" for v, c in top))

    console.print(tbl)

    # ── Score distribution ────────────────────────────────────────────────────
    s = sorted(all_scores)
    q1  = s[len(s) // 4]
    q3  = s[3 * len(s) // 4]
    stbl = Table(title=f"Score Distribution ({sim.METRIC_NAME})", show_header=False)
    stbl.add_column("Metric")
    stbl.add_column("Value", justify="right")
    for label, val in [("min", min(all_scores)), ("p25", q1),
                       ("median", statistics.median(all_scores)),
                       ("p75", q3), ("max", max(all_scores)),
                       ("stdev", statistics.stdev(all_scores))]:
        stbl.add_row(label, f"{val:.2f}")
    console.print(stbl)

    # ── Dominance check ───────────────────────────────────────────────────────
    wtbl = Table(title="Candidate Win Distribution (8 random strategies)", show_header=True)
    wtbl.add_column("Candidate")
    wtbl.add_column("Wins", justify="right")
    wtbl.add_column("Win %", justify="right")
    for i in range(len(candidates)):
        wins = win_counts.get(i, 0)
        wtbl.add_row(f"candidate_{i}", str(wins), f"{wins/n*100:.0f}%")
    console.print(wtbl)

    # ── Verdict ───────────────────────────────────────────────────────────────
    issues = []
    if max(all_scores) == min(all_scores):
        issues.append("All scores identical — simulate() may not depend on scenario state")
    elif statistics.stdev(all_scores) < abs(statistics.mean(all_scores)) * 0.05:
        issues.append("Very low score variance — check that scenario factors affect outcomes")
    top_pct = win_counts.most_common(1)[0][1] / n * 100
    if top_pct > 70:
        issues.append(f"One candidate wins {top_pct:.0f}% of rounds — likely a dominant strategy")
    zero_pct = sum(1 for sc in all_scores if sc == 0) / len(all_scores) * 100
    if zero_pct > 30:
        issues.append(f"{zero_pct:.0f}% of rounds score 0 — check simulate() edge cases")

    if issues:
        console.print("\n[yellow]Calibration warnings:[/yellow]")
        for issue in issues:
            console.print(f"  [yellow]! {issue}[/yellow]")
        console.print(f"\n[dim]Edit {args.domain}/simulation.py and re-run calibrate.[/dim]")
    else:
        console.print(f"\n[green]Calibration looks good.[/green]")
        console.print(f"  python run.py run --domain {args.domain} --batches 3 --rounds 50")


def cmd_validate(args):
    """Sanity-check simulation.py: imports, required exports, schema, sim output."""
    import os

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)

    errors   = []
    warnings = []
    ok       = []

    # 1. Import
    try:
        import importlib
        sim = importlib.import_module("simulation")
        ok.append("simulation.py imports cleanly")
    except Exception as e:
        errors.append(f"simulation.py import failed: {e}")
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 2. Required exports
    for name in ["simulate", "random_state", "CANDIDATE_SCHEMA", "METRIC_NAME"]:
        if hasattr(sim, name):
            ok.append(f"{name} exists")
        else:
            errors.append(f"Missing required export: {name}")

    if errors:
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 3. METRIC_NAME is a non-empty string
    if isinstance(sim.METRIC_NAME, str) and sim.METRIC_NAME.strip():
        ok.append(f"METRIC_NAME = '{sim.METRIC_NAME}'")
    else:
        errors.append("METRIC_NAME must be a non-empty string")

    # 4. CANDIDATE_SCHEMA is a valid dict with properties
    schema = sim.CANDIDATE_SCHEMA
    if isinstance(schema, dict) and "properties" in schema and schema["properties"]:
        n_params = len(schema["properties"])
        ok.append(f"CANDIDATE_SCHEMA has {n_params} parameter(s): {', '.join(schema['properties'].keys())}")
    else:
        errors.append("CANDIDATE_SCHEMA must be a dict with a non-empty 'properties' key")

    if errors:
        _print_validation(ok, warnings, errors)
        sys.exit(1)

    # 5. random_state() returns a dict
    try:
        state = sim.random_state()
        if isinstance(state, dict) and state:
            ok.append(f"random_state() returns dict with {len(state)} key(s): {', '.join(list(state.keys())[:5])}")
        else:
            errors.append("random_state() must return a non-empty dict")
    except Exception as e:
        errors.append(f"random_state() raised: {e}")

    # 6. Build a minimal valid candidate from schema and call simulate()
    try:
        candidate = {}
        for key, spec in schema.get("properties", {}).items():
            t = spec.get("type")
            if t == "number":
                lo = spec.get("minimum", 0.0)
                hi = spec.get("maximum", 1.0)
                candidate[key] = (lo + hi) / 2
            elif t == "integer":
                lo = spec.get("minimum", 0)
                hi = spec.get("maximum", 10)
                candidate[key] = (lo + hi) // 2
            elif "enum" in spec:
                candidate[key] = spec["enum"][0]
            else:
                candidate[key] = None

        state    = sim.random_state()
        score    = sim.simulate(candidate, state)

        if isinstance(score, (int, float)):
            ok.append(f"simulate() returns {sim.METRIC_NAME} = {score}")
        else:
            errors.append(f"simulate() must return a number, got {type(score).__name__}")
    except Exception as e:
        errors.append(f"simulate() raised: {e}")

    # 7. Score variety check — run 20 scenarios, warn if all identical
    try:
        import random as _random
        scores = [sim.simulate(candidate, sim.random_state()) for _ in range(20)]
        unique = len(set(round(s, 4) for s in scores))
        if unique < 5:
            warnings.append(f"simulate() returned only {unique} unique scores across 20 random scenarios — check calibration")
        else:
            ok.append(f"simulate() produces varied scores across scenarios ({unique}/20 unique)")
    except Exception:
        pass

    # 8. Check prompts exist
    for fname in ["brain.md", "extract.md", "director.md"]:
        p = domain_path / "prompts" / fname
        if p.exists():
            ok.append(f"prompts/{fname} exists")
        else:
            warnings.append(f"prompts/{fname} missing — needed for Stage 2 (--brain)")

    _print_validation(ok, warnings, errors)

    if errors:
        sys.exit(1)
    else:
        print(f"\nDomain {args.domain} looks good. Run:")
        print(f"  python run.py run --domain {args.domain} --batches 3 --rounds 50")


def cmd_export(args):
    """Export Stage 3 training data from tournament_log.jsonl."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    from engine_export import export_training_data
    export_training_data(domain_path)


def cmd_status(args):
    """Show domain state: playbook, champion, last run, log stats."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    # Playbook
    pb_path = domain_path / "playbook.jsonl"
    if pb_path.exists():
        playbook = [json.loads(l) for l in pb_path.read_text().splitlines() if l.strip()]
        top5 = sorted(playbook, key=lambda p: p.get("confidence", 0), reverse=True)[:5]
        tbl = Table(title=f"Playbook — {len(playbook)} principles", show_header=True)
        tbl.add_column("Topic")
        tbl.add_column("Conf", justify="right")
        tbl.add_column("Principle")
        for p in top5:
            tbl.add_row(
                p.get("topic", ""),
                f"{p.get('confidence', 0):.0%}",
                p.get("principle", "")[:60],
            )
        console.print(tbl)
    else:
        console.print("Playbook: not found")

    # Champion
    champion_path = domain_path / "champion_archetype.json"
    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            line = f"Champion: {champ.get('name', 'unknown')} — {champ.get('philosophy', '')[:70]}"
            console.print(f"\n[bold]{line}[/bold]")
        except Exception:
            pass

    # Retired topics
    rt_path = domain_path / "retired_topics.json"
    if rt_path.exists():
        try:
            retired = json.loads(rt_path.read_text())
            if retired:
                msg = f"Retired topics ({len(retired)}): {', '.join(retired)}"
                console.print(f"\n[dim]{msg}[/dim]")
        except Exception:
            pass

    # Last run
    last_run_path = domain_path / "last_run.json"
    if last_run_path.exists():
        try:
            lr = json.loads(last_run_path.read_text())
            lines = [
                f"Last run: {lr.get('timestamp', 'unknown')[:19]}",
                f"  Rounds: {lr.get('total_rounds', 0)}   Verdict: {lr.get('final_verdict', 'unknown')}",
            ]
            if lr.get("stop_reason"):
                lines.append(f"  Stop reason: {lr['stop_reason']}")
            console.print("\n" + "\n".join(lines))
        except Exception:
            pass

    # Tournament log + training data
    log_path = domain_path / "tournament_log.jsonl"
    td_path  = domain_path / "training_data.jsonl"
    if log_path.exists():
        try:
            n_rounds = sum(1 for l in log_path.read_text().splitlines() if l.strip())
            td_exists = td_path.exists()
            msg = f"tournament_log.jsonl: {n_rounds} rounds"
            if td_exists:
                n_td = sum(1 for l in td_path.read_text().splitlines() if l.strip())
                msg += f"   training_data.jsonl: {n_td} examples (ready)"
            else:
                msg += "   training_data.jsonl: not yet exported"
            console.print(f"\n[dim]{msg}[/dim]")
        except Exception:
            pass
