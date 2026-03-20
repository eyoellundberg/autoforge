"""
tools.py — Domain management commands: status, tail, pack, install, eval,
           import, export, train.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from utils import ENGINE_ROOT, load_env, load_sim, normalize_confidence, load_jsonl


def cmd_status(args):
    """Show domain state: playbook, champion, last run, log stats."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    pb_path = domain_path / "playbook.jsonl"
    if pb_path.exists():
        playbook = [json.loads(l) for l in pb_path.read_text().splitlines() if l.strip()]
        top5 = sorted(playbook, key=lambda p: normalize_confidence(p.get("confidence", 0)), reverse=True)[:5]
        print(f"\nPlaybook — {len(playbook)} principles (top 5)")
        print(f"  {'Topic':<22} {'Conf':>5}  Principle")
        print(f"  {'─'*72}")
        for p in top5:
            print(f"  {p.get('topic',''):<22} {normalize_confidence(p.get('confidence',0)):>4.0%}  {p.get('principle','')[:60]}")
    else:
        print("Playbook: not found")

    champion_path = domain_path / "champion_archetype.json"
    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            print(f"\nChampion: {champ.get('name', 'unknown')} — {champ.get('philosophy', '')[:70]}")
        except Exception:
            pass

    rt_path = domain_path / "retired_topics.json"
    if rt_path.exists():
        try:
            retired = json.loads(rt_path.read_text())
            if retired:
                print(f"\nRetired topics ({len(retired)}): {', '.join(retired)}")
        except Exception:
            pass

    last_run_path = domain_path / "last_run.json"
    if last_run_path.exists():
        try:
            lr = json.loads(last_run_path.read_text())
            print(f"\nLast run: {lr.get('timestamp', 'unknown')[:19]}")
            print(f"  Rounds: {lr.get('total_rounds', 0)}   Verdict: {lr.get('final_verdict', 'unknown')}")
            if lr.get("stop_reason"):
                print(f"  Stop reason: {lr['stop_reason']}")
        except Exception:
            pass

    log_path   = domain_path / "tournament_log.jsonl"
    td_path    = domain_path / "training_data.jsonl"
    pref_path  = domain_path / "training_preferences.jsonl"
    thresh_path = domain_path / "abstain_threshold.json"
    if log_path.exists():
        try:
            n_rounds = sum(1 for l in log_path.read_text().splitlines() if l.strip())
            print(f"\ntournament_log.jsonl: {n_rounds} rounds")
            if td_path.exists():
                n_td = sum(1 for l in td_path.read_text().splitlines() if l.strip())
                print(f"training_data.jsonl:  {n_td} examples (ready)")
            else:
                print(f"training_data.jsonl:  not yet exported")
            if pref_path.exists():
                n_pref = sum(1 for l in pref_path.read_text().splitlines() if l.strip())
                print(f"training_preferences.jsonl: {n_pref} pairs")
            if thresh_path.exists():
                threshold = json.loads(thresh_path.read_text()).get("threshold")
                print(f"abstain_threshold: {threshold}")
        except Exception:
            pass


def cmd_tail(args):
    """Live snapshot of an in-progress run."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    checkpoint_path = domain_path / "run_checkpoint.json"
    ckpt = None
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
        except Exception:
            pass

    log_path = domain_path / "tournament_log.jsonl"
    recent_scores = []
    total_rounds  = 0
    if log_path.exists():
        lines = [l for l in log_path.read_text().splitlines() if l.strip()]
        total_rounds = len(lines)
        for line in lines[-50:]:
            try:
                score = json.loads(line).get("score")
                if score is not None:
                    recent_scores.append(float(score))
            except Exception:
                pass

    print(f"\nTail — {args.domain}")
    print(f"{'─'*40}")

    if ckpt:
        stage    = "Stage 2 (AI)" if ckpt.get("use_brain") else "Stage 1 (evolutionary)"
        pb_sizes = ckpt.get("playbook_sizes", [])
        print(f"Batch:    {ckpt['global_batch']}  (year {ckpt['year_num']})")
        print(f"Stage:    {stage}")
        print(f"Playbook: {pb_sizes[-1] if pb_sizes else '?'} principles")
        pa = ckpt.get("prior_analysis") or {}
        if pa.get("verdict"):
            print(f"Verdict:  {pa['verdict']}")
        if pa.get("next_batch_focus"):
            print(f"Focus:    {pa['next_batch_focus'][:72]}")
        for row in ckpt.get("all_batch_rows", [])[-6:]:
            arrow = "↑" if row["trend"] == "up" else ("↓" if row["trend"] == "down" else "─")
            print(f"  batch {row['batch']:3d}  avg {row['avg']:6.2f}  best {row['best']:6.2f}  {arrow}")
    else:
        print("No active run checkpoint.")

    print(f"\nRounds logged: {total_rounds}")
    if recent_scores:
        avg = sum(recent_scores) / len(recent_scores)
        print(f"Last 50:  avg {avg:.2f}  min {min(recent_scores):.2f}  max {max(recent_scores):.2f}")
    print()


def cmd_pack(args):
    """Bundle a domain into a shareable .zip pack."""
    import zipfile

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    pack_path = domain_path / "pack.json"
    if pack_path.exists():
        pack = json.loads(pack_path.read_text())
    else:
        sim_path = domain_path / "simulation.py"
        metric = "score"
        if sim_path.exists():
            for line in sim_path.read_text().splitlines():
                if "METRIC_NAME" in line and "=" in line:
                    metric = line.split("=", 1)[1].strip().strip('"\'')
                    break
        pack = {
            "name": args.domain, "version": "1.0.0", "author": "",
            "description": "", "metric": metric,
            "autoforge_version": "1.0", "evals": "evals/scenarios.jsonl",
        }
        pack_path.write_text(json.dumps(pack, indent=2) + "\n")
        print(f"  Created pack.json (edit version/author/description before sharing)")

    include = [
        "simulation.py", "world_model.md", "mission.md", "pack.json",
        "hypotheses.json", "prompts/brain.md", "prompts/extract.md",
        "prompts/director.md", "playbook.jsonl", "champion_archetype.json",
        "top_candidates.json", "abstain_threshold.json",
    ]
    evals_path = domain_path / "evals"
    if evals_path.exists():
        for f in evals_path.iterdir():
            include.append(f"evals/{f.name}")

    name     = pack.get("name", args.domain)
    version  = pack.get("version", "1.0.0")
    zip_name = f"{name}-{version}.zip"
    zip_path = ENGINE_ROOT / zip_name

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            src = domain_path / rel
            if src.exists():
                zf.write(src, f"{name}/{rel}")

    size_kb = zip_path.stat().st_size // 1024
    print(f"\nPacked {args.domain} → {zip_name}  ({size_kb} KB)")
    for f in [r for r in include if (domain_path / r).exists()]:
        print(f"  {f}")
    print(f"\nShare {zip_name} — install with: autoforge install {zip_name}")


def cmd_install(args):
    """Install a domain pack from a .zip file."""
    import zipfile
    zip_path = Path(args.pack) if Path(args.pack).is_absolute() else ENGINE_ROOT / args.pack

    if not zip_path.exists():
        print(f"Pack file not found: {zip_path}")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if not names:
            print("Empty zip file.")
            sys.exit(1)
        domain_name = names[0].split("/")[0]
        domain_path = ENGINE_ROOT / domain_name

        if domain_path.exists() and not getattr(args, "force", False):
            print(f"Domain '{domain_name}' already exists. Use --force to overwrite.")
            sys.exit(1)

        domain_path.mkdir(exist_ok=True)
        zf.extractall(ENGINE_ROOT)

    nested = domain_path / domain_name
    if nested.exists():
        for item in nested.iterdir():
            shutil.move(str(item), str(domain_path / item.name))
        nested.rmdir()

    pack_file = domain_path / "pack.json"
    if pack_file.exists():
        pack = json.loads(pack_file.read_text())
        print(f"\nInstalled: {pack.get('name', domain_name)} v{pack.get('version', '?')}")
        print(f"  {pack.get('description', '')}")
        print(f"  Metric: {pack.get('metric', '?')}")
    else:
        print(f"\nInstalled: {domain_name}")

    print(f"\nNext steps:")
    print(f"  autoforge calibrate --domain {domain_name}")
    print(f"  autoforge run --domain {domain_name} --batches 5 --rounds 100")
    print(f"  autoforge eval --domain {domain_name}")


def cmd_eval(args):
    """Run the champion strategy against eval scenarios and report pass/fail."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    evals_path = domain_path / "evals" / "scenarios.jsonl"
    if not evals_path.exists():
        print(f"No evals found at {evals_path}")
        print("Add eval scenarios in evals/scenarios.jsonl:")
        print('  {"id": "test_1", "state": {...}, "description": "...", "min_score": 0}')
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
        sys.exit(1)

    strategies = []
    champion_path = domain_path / "champion_archetype.json"
    top_path      = domain_path / "top_candidates.json"

    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            strategies = [{"name": champ.get("name", "champion"), "strategy": champ["strategy"]}]
        except Exception:
            pass
    if not strategies and top_path.exists():
        try:
            top = json.loads(top_path.read_text())
            strategies = [{"name": t.get("name", f"top_{i}"), "strategy": t["strategy"]} for i, t in enumerate(top[:4])]
        except Exception:
            pass
    if not strategies:
        print("No champion_archetype.json or top_candidates.json — run some batches first.")
        sys.exit(1)

    scenarios = load_jsonl(evals_path, skip_comments=True)
    if not scenarios:
        print("evals/scenarios.jsonl is empty or has no valid JSON lines.")
        sys.exit(1)

    print(f"\nEval Results — {args.domain} ({len(scenarios)} scenarios)")
    print(f"  {'ID':<16} {'Description':<42} {'Score':>7} {'Min':>6}  Pass")
    print(f"  {'─'*80}")

    passed = 0
    for scen in scenarios:
        state       = scen.get("state", {})
        min_score   = scen.get("min_score", 0)
        description = scen.get("description", "")[:40]
        scen_id     = scen.get("id", "?")

        best_score = max(sim.simulate(s["strategy"], state) for s in strategies)
        ok = best_score >= min_score
        if ok:
            passed += 1
        print(f"  {scen_id:<16} {description:<42} {best_score:>7.2f} {min_score:>6.2f}  {'✓' if ok else '✗'}")

    pct = passed / len(scenarios) * 100
    print(f"\n{passed}/{len(scenarios)} passed ({pct:.0f}%)")
    if pct < 80:
        print("Run more batches to improve — then re-eval.")
    if passed < len(scenarios):
        sys.exit(1)


def cmd_export(args):
    """Export Stage 3 training data from tournament_log.jsonl."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    from export import export_training_data
    export_training_data(domain_path)


def cmd_generate_evals(args):
    """Auto-generate eval scenarios from world_model.md."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    from validate import generate_evals
    scenarios = generate_evals(domain_path, n=args.n)
    if scenarios:
        print(f"\nGenerated {len(scenarios)} eval scenarios in {args.domain}/evals/scenarios.jsonl")
        for s in scenarios:
            print(f"  {s['id']}: {s['description'][:60]}")
        print(f"\nRun evals:  autoforge eval --domain {args.domain}")


def _generate_specialist(domain_path: Path, domain_name: str, domain_type: str,
                         feature_cols: list, state_keys: list, param_keys: list,
                         param_ranges: dict, abstain_threshold: float, r2: float,
                         n_examples: int):
    """Generate the standalone specialist/ folder — copy templates + write config."""
    spec_dir = domain_path / "specialist"
    spec_dir.mkdir(exist_ok=True)

    config = {
        "domain":            domain_name,
        "domain_type":       domain_type,
        "features":          feature_cols,
        "state_keys":        state_keys,
        "params":            param_keys,
        "param_ranges":      param_ranges,
        "abstain_threshold": abstain_threshold,
        "trained_on":        n_examples,
        "r2":                round(r2, 4),
        "version":           datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    (spec_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    src_model = domain_path / "model.json"
    if src_model.exists():
        shutil.copy2(src_model, spec_dir / "model.json")

    template_spec = ENGINE_ROOT / "template" / "specialist"
    for fname in ("predict.py", "retrain.py"):
        src = template_spec / fname
        if src.exists():
            shutil.copy2(src, spec_dir / fname)

    return spec_dir


def _train_xgboost(domain_path: Path, domain_name: str, *, verbose: bool = False):
    """
    Shared XGBoost training logic: train model, compute R², detect columns,
    generate specialist/. Returns (r2, n_examples) or None on failure.
    """
    import importlib

    try:
        import xgboost as xgb
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"  XGBoost not installed — run: pip install xgboost pandas numpy")
        return None

    csv_path = domain_path / "training_features.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("score", "score_margin", "uncertain")]
    X, y = df[feature_cols].values, df["score"].values

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
    train_kwargs = {}
    if verbose:
        train_kwargs = {"evals": [(dtrain, "train")], "verbose_eval": 50}
    model = xgb.train(params, dtrain, num_boost_round=200, **train_kwargs)
    model.save_model(str(domain_path / "model.json"))

    preds = model.predict(dtrain)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Detect state vs param columns
    state_keys, param_keys = feature_cols, []
    try:
        sys.path.insert(0, str(domain_path))
        sim = importlib.import_module("simulation")
        schema_props = list(getattr(sim, "CANDIDATE_SCHEMA", {}).get("properties", {}).keys())
        param_keys = [c for c in feature_cols if c in schema_props]
        state_keys = [c for c in feature_cols if c not in schema_props]
        sys.path.pop(0)
        if "simulation" in sys.modules:
            del sys.modules["simulation"]
    except Exception:
        pass

    param_ranges = {}
    for p in param_keys:
        if p in df.columns:
            col = pd.to_numeric(df[p], errors="coerce").dropna()
            if len(col) > 0:
                param_ranges[p] = [float(col.min()), float(col.max())]

    abstain_threshold = 0.0
    threshold_path = domain_path / "abstain_threshold.json"
    if threshold_path.exists():
        try:
            abstain_threshold = json.loads(threshold_path.read_text()).get("threshold", 0.0)
        except Exception:
            pass

    _generate_specialist(
        domain_path, domain_name, "numerical",
        feature_cols, state_keys, param_keys,
        param_ranges, abstain_threshold, r2, len(y),
    )

    return r2, len(y)


def cmd_train(args):
    """Train a Stage 3 model from exported training data."""
    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    from export import _detect_domain_type, export_training_data
    domain_type = _detect_domain_type(domain_path)

    csv_path   = domain_path / "training_features.csv"
    jsonl_path = domain_path / "training_data.jsonl"

    if not csv_path.exists() and not jsonl_path.exists():
        print("No training data found — running export first...\n")
        export_training_data(domain_path)

    if domain_type == "numerical":
        if not csv_path.exists():
            print(f"No training_features.csv — run: autoforge export --domain {args.domain}")
            sys.exit(1)

        print(f"Training XGBoost on {csv_path.name}...")
        result = _train_xgboost(domain_path, args.domain, verbose=True)
        if result is None:
            sys.exit(1)
        r2, n_examples = result

        print(f"\nR²: {r2:.3f}  ({n_examples} examples)")
        print(f"\nDeployable:  {args.domain}/specialist/")
        print(f"  predict.py   — from specialist.predict import predict, record")
        print(f"  retrain.py   — python retrain.py  (retrains on real outcomes)")
        print(f"  model.json   — XGBoost weights")
        print(f"  config.json  — features, params, threshold")

    else:
        if not jsonl_path.exists():
            print(f"No training_data.jsonl — run: autoforge export --domain {args.domain}")
            sys.exit(1)

        n  = sum(1 for l in jsonl_path.read_text().splitlines() if l.strip())
        dn = args.domain
        print(f"Language domain — {n} training examples")
        print(f"\nFine-tune (Apple Silicon):")
        print(f"  mlx_lm.lora --model mlx-community/Qwen3-4B-4bit \\")
        print(f"               --data {dn}/ --train --iters 1000")
        print(f"\nFuse + Ollama:")
        print(f"  mlx_lm.fuse --model mlx-community/Qwen3-4B-4bit \\")
        print(f"               --adapter-path {dn}/adapters --save-path {dn}/fused-model")
        print(f'  ollama create {dn.lower()} -f - <<EOF\nFROM ./{dn}/fused-model\nEOF')


def cmd_import(args):
    """
    Import real production decisions into the domain's training data.
    Reads a JSONL file: {"state": {...}, "decision": {...}, "outcome": float}
    """
    import statistics

    domain_path = ENGINE_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    input_path = Path(args.file)
    if not input_path.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
        sys.exit(1)

    expected_state_keys    = set(sim.random_state().keys())
    expected_decision_keys = set(sim.CANDIDATE_SCHEMA.get("properties", {}).keys())
    metric_name = sim.METRIC_NAME

    log_path  = domain_path / "tournament_log.jsonl"
    prod_path = domain_path / "production_log.jsonl"

    max_round = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                max_round = max(max_round, json.loads(line).get("round", 0))
            except Exception:
                pass

    raw_lines = [l.strip() for l in input_path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    records = []
    skipped = []
    for i, line in enumerate(raw_lines):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            skipped.append(f"line {i+1}: invalid JSON")
            continue

        state    = rec.get("state", {})
        decision = rec.get("decision", {})
        outcome  = rec.get("outcome")

        missing_state    = expected_state_keys - set(state.keys())
        missing_decision = expected_decision_keys - set(decision.keys())

        if missing_state:
            skipped.append(f"line {i+1}: missing state keys: {missing_state}")
            continue
        if missing_decision:
            skipped.append(f"line {i+1}: missing decision keys: {missing_decision}")
            continue
        if outcome is None:
            skipped.append(f"line {i+1}: missing 'outcome' field")
            continue
        try:
            outcome = float(outcome)
        except (TypeError, ValueError):
            skipped.append(f"line {i+1}: outcome must be a number")
            continue

        records.append({"state": state, "decision": decision, "outcome": outcome})

    print(f"\nImport: {args.file} → {args.domain}")
    print(f"  {len(records)} valid  /  {len(skipped)} skipped")
    if skipped:
        for msg in skipped[:5]:
            print(f"  [skip] {msg}")
        if len(skipped) > 5:
            print(f"  ... and {len(skipped)-5} more")

    if not records:
        print("Nothing to import.")
        return

    outcomes = [r["outcome"] for r in records]
    print(f"  Score range: {min(outcomes):.2f} – {max(outcomes):.2f}  median {statistics.median(outcomes):.2f}")

    sim_states = [sim.random_state() for _ in range(min(len(records) * 2, 200))]
    print(f"\n  Distribution comparison (imported vs simulated):")
    for key in sorted(expected_state_keys):
        imported_vals = [r["state"].get(key) for r in records if isinstance(r["state"].get(key), (int, float))]
        sim_vals      = [s.get(key) for s in sim_states if isinstance(s.get(key), (int, float))]
        if imported_vals and sim_vals:
            print(f"    {key}:  imported median {statistics.median(imported_vals):.2f}  |  sim median {statistics.median(sim_vals):.2f}")

    with open(prod_path, "a") as f:
        for rec in records:
            f.write(json.dumps({"source": "production", **rec}) + "\n")

    with open(log_path, "a") as f:
        for i, rec in enumerate(records):
            f.write(json.dumps({
                "round":  max_round + i + 1,
                "score":  rec["outcome"],
                "metric": metric_name,
                "winner": rec["decision"],
                "state":  rec["state"],
                "source": "production",
            }) + "\n")

    print(f"\nImported {len(records)} records.")
    print(f"  production_log.jsonl — provenance trail")
    print(f"  tournament_log.jsonl — {len(records)} records added to Stage 3 training data")
    print(f"\nRun 'autoforge export --domain {args.domain}' to include in Stage 3.")
