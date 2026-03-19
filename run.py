"""
run.py — Tournament orchestration loop (cmd_run).

Also serves as the CLI entry point when invoked directly:
  python run.py Domain "description"
  python run.py run --domain Domain --batches 8 --rounds 200
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from utils import (
    ENGINE_ROOT,
    load_env,
    load_hypotheses,
    save_hypotheses,
    retire_principles,
    append_thinking_log,
    git_commit_batch,
)
from api import get_ai_backend
from director import call_director


def _auto_train_numerical(domain_path, domain_name):
    """Train XGBoost + generate specialist/ automatically on saturation."""
    try:
        import xgboost as xgb
        import pandas as pd
        import numpy as np
    except ImportError:
        print("  XGBoost not installed — run: pip install xgboost pandas numpy")
        print(f"  Then: autoforge train --domain {domain_name}")
        return

    csv_path = domain_path / "training_features.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ("score", "score_margin", "uncertain")]
    X, y = df[feature_cols].values, df["score"].values

    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)
    params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
    model = xgb.train(params, dtrain, num_boost_round=200)
    model.save_model(str(domain_path / "model.json"))

    preds = model.predict(dtrain)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Detect state vs param columns
    state_keys, param_keys = feature_cols, []
    try:
        import importlib
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

    from tools import _generate_specialist
    _generate_specialist(
        domain_path, domain_name, "numerical",
        feature_cols, state_keys, param_keys,
        param_ranges, abstain_threshold, r2, len(y),
    )

    print(f"\nR²: {r2:.3f}  ({len(y)} examples)")
    print(f"\nDeployable:  {domain_name}/specialist/")
    print(f"  predict.py   — from specialist.predict import predict, record")
    print(f"  retrain.py   — python retrain.py  (retrains on real outcomes)")
    print(f"  model.json   — XGBoost weights")
    print(f"  config.json  — features, params, threshold")


def _default_analysis(hints=None, focus=""):
    """Return a default analysis dict for when the director is unavailable."""
    return {
        "verdict": "exploring",
        "hints": hints or [],
        "retire_principles": [],
        "next_batch_focus": focus or "director unavailable",
        "observations": [],
        "principles_gaining_confidence": [],
        "concerns": [],
        "mistakes_to_note": [],
        "simulation_fix_suggestions": [],
        "hypotheses_tested": [],
        "hypotheses_confirmed": [],
        "hypotheses_open": [],
        "simulation_patch_needed": False,
        "simulation_patch_rationale": "",
        "schema_evolution": {"add_parameters": [], "remove_parameters": []},
    }


def cmd_run(args):
    """Run the tournament in batches with optional AI direction."""
    domain_path = ENGINE_ROOT / args.domain

    if not domain_path.exists():
        print(f"Domain folder not found: {domain_path}")
        sys.exit(1)

    if getattr(args, "manual_ai", False):
        os.environ["AUTOFORGE_AI_BACKEND"] = "manual"

    load_env(domain_path)

    from tournament import run_batch

    thinking_log = domain_path / "thinking_log.md"
    if not thinking_log.exists():
        thinking_log.write_text(
            f"# Thinking Log — {args.domain}\n\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
            f"|  {args.batches} batches × {args.rounds} rounds\n\n---\n"
        )

    use_brain      = args.brain
    auto_mode      = getattr(args, "auto", False)
    self_evolve    = getattr(args, "self_evolve", False)
    ground_enabled = getattr(args, "ground", False)
    brain_label    = "Stage 2 — AI archetypes" if use_brain else ("auto" if auto_mode else "Stage 1 — procedural")
    if self_evolve:
        brain_label += " + self-evolve"
    if ground_enabled:
        brain_label += " + ground"
    backend_label = get_ai_backend()

    print(f"\n{'='*60}")
    print(f"Autoforge — {args.domain}  [{brain_label}]")
    if use_brain or auto_mode:
        print(f"AI backend: {backend_label}")
    print(f"{args.years} year(s) x {args.batches} batches x {args.rounds} rounds = {args.years * args.batches * args.rounds} total")
    print(f"{'='*60}\n")

    generation_offset     = 0
    year_avg_scores       = []
    prior_analysis        = None
    playbook_sizes        = []
    stop_reason           = None
    all_batch_rows        = []
    auto_consecutive_flat = 0
    hints                 = []
    adversarial_states    = []
    start_year            = 1
    start_batch           = 1
    run_id                = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Sim hash — warn if simulation.py changed since last run
    import hashlib as _hashlib
    sim_path = domain_path / "simulation.py"
    if sim_path.exists():
        sim_hash = _hashlib.md5(sim_path.read_bytes()).hexdigest()[:8]
        pack_path = domain_path / "pack.json"
        pack_data = {}
        if pack_path.exists():
            try:
                pack_data = json.loads(pack_path.read_text())
            except Exception:
                pass
        stored_hash = pack_data.get("sim_hash")
        if stored_hash and stored_hash != sim_hash:
            print(f"[warn] simulation.py changed since last run ({stored_hash} → {sim_hash})")
            print("       Previous tournament_log.jsonl was generated by a different sim.")
            print("       Delete tournament_log.jsonl to start fresh, or continue if intentional.\n")
        pack_data["sim_hash"] = sim_hash
        if pack_path.exists():
            pack_path.write_text(json.dumps(pack_data, indent=2) + "\n")

    checkpoint_path = domain_path / "run_checkpoint.json"
    if checkpoint_path.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text())
            if ckpt.get("domain") == args.domain:
                generation_offset     = ckpt["generation_offset"]
                playbook_sizes        = ckpt["playbook_sizes"]
                prior_analysis        = ckpt.get("prior_analysis")
                auto_consecutive_flat = ckpt.get("auto_consecutive_flat", 0)
                use_brain             = ckpt.get("use_brain", use_brain)
                hints                 = ckpt.get("hints", [])
                all_batch_rows        = ckpt.get("all_batch_rows", [])
                year_avg_scores       = ckpt.get("year_avg_scores", [])
                start_year            = ckpt["year_num"]
                start_batch           = ckpt["batch_num"] + 1
                print(f"Resuming from batch {ckpt['global_batch']} (delete run_checkpoint.json to start fresh)\n")
        except Exception:
            pass

    for year_num in range(start_year, args.years + 1):
        all_results = []

        if args.years > 1:
            print(f"\n{'─'*60}")
            print(f"Year {year_num} / {args.years}  — playbook carries over from prior year")
            print(f"{'─'*60}\n")
            with open(thinking_log, "a") as f:
                f.write(f"\n# -- Year {year_num} -- {datetime.now().strftime('%Y-%m-%d %H:%M')} --\n\n")

        for batch_num in range(start_batch if year_num == start_year else 1, args.batches + 1):
            global_batch = (year_num - 1) * args.batches + batch_num

            # Auto-mode: promote to Stage 2 if playbook flatlined for 2 consecutive batches
            if auto_mode and not use_brain and auto_consecutive_flat >= 2:
                use_brain = True
                print(f"[auto] Playbook flat for 2 batches — promoting to Stage 2 (AI archetypes)\n")
                with open(thinking_log, "a") as f:
                    f.write(f"\n*[Auto] Stage 1 saturated at batch {global_batch} — promoted to Stage 2 (AI archetypes).*\n\n")

            print(f"[Y{year_num} Batch {batch_num}/{args.batches}] running {args.rounds} rounds", end="", flush=True)

            try:
                result = run_batch(
                    domain_path,
                    args.rounds,
                    generation_offset=generation_offset,
                    hints=hints,
                    use_brain=use_brain,
                    workers=args.workers,
                    run_id=run_id,
                    adversarial_states=adversarial_states,
                )
            except Exception as e:
                print(f"\n  [batch error: {e}] — skipping to next batch")
                with open(thinking_log, "a") as f:
                    f.write(f"\n*Batch {global_batch} failed: {e}*\n\n")
                generation_offset += args.rounds
                continue

            all_results.append(result)
            generation_offset += args.rounds

            current_size = result["playbook_size"]
            playbook_sizes.append(current_size)

            # Auto-mode: track consecutive flat batches
            if auto_mode and not use_brain:
                if len(playbook_sizes) >= 2 and playbook_sizes[-1] == playbook_sizes[-2]:
                    auto_consecutive_flat += 1
                else:
                    auto_consecutive_flat = 0

            trend_char = "+" if result["trend_pct"] >= 0 else ""
            print(f"  ->  avg {result['avg_score']}  best {result['best_score']}  trend {trend_char}{result['trend_pct']:.1f}%")

            if use_brain:
                print(f"  director analyzing...", end="", flush=True)
                try:
                    analysis = call_director(global_batch, result, prior_analysis, all_results, domain_path, playbook_sizes)
                    append_thinking_log(thinking_log, global_batch, result, analysis)
                    verdict = analysis["verdict"]
                    print(f"  [{verdict}]")
                    print(f"  -> {analysis['next_batch_focus'][:80]}")
                    if analysis["concerns"]:
                        print(f"  ! {analysis['concerns'][0]}")
                    if verdict == "needs_calibration" and analysis.get("simulation_fix_suggestions"):
                        print("\nSimulation fix suggestions:")
                        for fix in analysis["simulation_fix_suggestions"]:
                            print(f"  * {fix}")
                        print(f"  Edit {args.domain}/simulation.py and re-run calibrate.\n")
                except Exception as e:
                    print(f"  [director error: {e}] — continuing")
                    analysis = _default_analysis(hints=hints, focus="director unavailable — continuing previous hints")
                    verdict = "exploring"
            else:
                analysis = _default_analysis(focus="Stage 1 — use --brain or --auto to enable director")
                verdict = "exploring"
                print(f"  [Stage 1]")
            print()

            if len(all_batch_rows) > 0:
                prev_avg = all_batch_rows[-1]["avg"]
                trend_sym = "up" if result["avg_score"] > prev_avg else ("down" if result["avg_score"] < prev_avg else "flat")
            else:
                trend_sym = "--"
            all_batch_rows.append({
                "batch": global_batch,
                "avg":   result["avg_score"],
                "best":  result["best_score"],
                "trend": trend_sym,
            })

            # Reality grounding
            if ground_enabled and use_brain:
                print(f"  grounding...", end="", flush=True)
                try:
                    from ground import run_grounding, append_grounding_to_thinking_log
                    skip_cal = (global_batch % 3 != 0)
                    ground_report = run_grounding(
                        domain_path, result, analysis,
                        skip_calibration_audit=skip_cal,
                    )
                    append_grounding_to_thinking_log(thinking_log, global_batch, ground_report)

                    n_anomalies = len(ground_report.get("anomalies", []))
                    cal_verdict = ground_report.get("calibration_verdict", "skipped")
                    print(f"  [{cal_verdict}, {n_anomalies} anomalies]")

                    ground_hyp = ground_report.get("new_hypotheses", [])
                    if ground_hyp:
                        analysis.setdefault("hypotheses_open", []).extend(ground_hyp)

                    ground_concerns = ground_report.get("sim_concerns", [])
                    if ground_concerns:
                        analysis.setdefault("concerns", []).extend(ground_concerns)

                    if cal_verdict == "miscalibrated" and self_evolve:
                        analysis["simulation_patch_needed"] = True
                        if not analysis.get("simulation_patch_rationale"):
                            analysis["simulation_patch_rationale"] = "Grounding layer flagged miscalibration: " + "; ".join(ground_concerns[:3])

                except Exception as e:
                    print(f"  [grounding error: {e}]")

            hints          = analysis["hints"]
            prior_analysis = analysis

            retire_principles(analysis, domain_path)

            # Hypothesis tracking
            if use_brain:
                prior_hypotheses = load_hypotheses(domain_path)
                save_hypotheses(domain_path, analysis, prior_hypotheses)

                wm_path = domain_path / "world_model.md"
                if wm_path.exists():
                    wm_text = wm_path.read_text()
                    hypotheses_data = load_hypotheses(domain_path)
                    hyp_lines = []
                    if hypotheses_data.get("confirmed"):
                        hyp_lines.append("Confirmed:")
                        hyp_lines.extend(f"- {h}" for h in hypotheses_data["confirmed"])
                        hyp_lines.append("")
                    if hypotheses_data.get("open"):
                        hyp_lines.append("Open:")
                        hyp_lines.extend(f"- {h}" for h in hypotheses_data["open"])
                    else:
                        hyp_lines.append("- (none open)")
                    hyp_content = "\n".join(hyp_lines)

                    if "<!-- hypotheses-start -->" in wm_text:
                        wm_text = re.sub(
                            r"<!-- hypotheses-start -->.*?<!-- hypotheses-end -->",
                            f"<!-- hypotheses-start -->\n{hyp_content}\n<!-- hypotheses-end -->",
                            wm_text,
                            flags=re.DOTALL,
                        )
                    else:
                        wm_text = re.sub(
                            r"(## Current Hypotheses\n).*?(?=\n## |\Z)",
                            f"\\1{hyp_content}\n",
                            wm_text,
                            flags=re.DOTALL,
                        )
                    wm_path.write_text(wm_text)

            # Self-evolve: apply simulation patches if flagged
            if self_evolve and use_brain and analysis.get("simulation_patch_needed"):
                from evolve import evolve_simulation
                patched = evolve_simulation(
                    domain_path,
                    rationale=analysis.get("simulation_patch_rationale", ""),
                    fix_suggestions=analysis.get("simulation_fix_suggestions", []),
                )
                if patched:
                    if "simulation" in sys.modules:
                        del sys.modules["simulation"]
                    pack_path = domain_path / "pack.json"
                    if pack_path.exists():
                        import hashlib as _hashlib
                        new_hash = _hashlib.md5((domain_path / "simulation.py").read_bytes()).hexdigest()[:8]
                        pack_data = json.loads(pack_path.read_text())
                        pack_data["sim_hash"] = new_hash
                        pack_path.write_text(json.dumps(pack_data, indent=2) + "\n")

            # Self-evolve: apply schema evolution if proposed
            if self_evolve and use_brain:
                schema_evo = analysis.get("schema_evolution", {})
                if schema_evo.get("add_parameters") or schema_evo.get("remove_parameters"):
                    from evolve import evolve_schema
                    evolved = evolve_schema(domain_path, schema_evo)
                    if evolved and "simulation" in sys.modules:
                        del sys.modules["simulation"]

            git_commit_batch(args.domain, domain_path, global_batch, result, analysis)

            # Generate adversarial scenarios every 2 Stage 2 batches
            if use_brain and global_batch % 2 == 0:
                try:
                    from brain import call_adversarial
                    champion = None
                    champ_path = domain_path / "champion_archetype.json"
                    if champ_path.exists():
                        champion = json.loads(champ_path.read_text())
                    adversarial_states = call_adversarial(
                        domain_path, n=args.rounds // 5,
                        context_mix=result.get("context_mix"),
                        champion=champion,
                    )
                    if adversarial_states:
                        print(f"  [adversarial] {len(adversarial_states)} targeted scenarios queued for next batch")
                except Exception:
                    adversarial_states = []
            elif not use_brain:
                adversarial_states = []

            checkpoint_path.write_text(json.dumps({
                "domain":               args.domain,
                "generation_offset":    generation_offset,
                "global_batch":         global_batch,
                "batch_num":            batch_num,
                "year_num":             year_num,
                "use_brain":            use_brain,
                "hints":                hints,
                "prior_analysis":       prior_analysis,
                "playbook_sizes":       playbook_sizes,
                "auto_consecutive_flat": auto_consecutive_flat,
                "all_batch_rows":       all_batch_rows,
                "year_avg_scores":      year_avg_scores,
                "timestamp":            datetime.now().isoformat(),
            }, indent=2))

            if verdict == "reward_hacking":
                print("! Reward hacking flagged — stopping run.")
                fixes = analysis.get("simulation_fix_suggestions", [])
                if fixes:
                    print("\nSimulation fix suggestions:")
                    for fix in fixes:
                        print(f"  * {fix}")
                    print(f"\n  Edit {args.domain}/simulation.py, then re-run calibrate.")
                else:
                    print(f"  Check thinking_log.md for details.")
                stop_reason = "reward_hacking"
                year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))
                break

            if verdict == "saturated":
                print("Playbook saturated — engine has learned everything this sim can teach.\n")
                try:
                    from export import export_training_data, _detect_domain_type
                    export_training_data(domain_path)
                    domain_type = _detect_domain_type(domain_path)
                    if domain_type == "numerical":
                        _auto_train_numerical(domain_path, args.domain)
                except Exception as e:
                    print(f"  [auto-train failed: {e}]")
                    print(f"  Run manually: autoforge train --domain {args.domain}")
                stop_reason = "saturated"
                year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))
                break

        else:
            year_avg_scores.append(round(sum(r["avg_score"] for r in all_results) / len(all_results), 2))

        if stop_reason in ("reward_hacking", "saturated"):
            break

    # Clean up checkpoint on normal completion
    if checkpoint_path.exists() and stop_reason not in ("reward_hacking",):
        checkpoint_path.unlink()

    # Write last_run.json
    final_verdict = prior_analysis["verdict"] if prior_analysis else "unknown"
    last_run = {
        "timestamp":       datetime.now().isoformat(),
        "domain":          args.domain,
        "total_rounds":    generation_offset,
        "year_avg_scores": year_avg_scores,
        "final_verdict":   final_verdict,
        "stop_reason":     stop_reason,
    }
    (domain_path / "last_run.json").write_text(json.dumps(last_run, indent=2))

    # Terminal bell
    print("\a", end="", flush=True)

    # Summary table
    print(f"\nRun Summary — {args.domain}")
    print(f"  {'Batch':<7} {'Avg':>7} {'Best':>7}  Trend")
    print(f"  {'─'*32}")
    for row in all_batch_rows:
        arrow = "up" if row["trend"] == "up" else ("down" if row["trend"] == "down" else row["trend"])
        print(f"  {row['batch']:<7} {row['avg']:>7} {row['best']:>7}  {arrow}")

    print(f"\n{'─'*50}")
    print(f"Final State")
    print(f"  Verdict:      {final_verdict}")
    if stop_reason:
        print(f"  Stop reason:  {stop_reason}")
    if prior_analysis and prior_analysis.get("concerns"):
        print(f"  Top concern:  {prior_analysis['concerns'][0]}")
    print(f"  Thinking log: {thinking_log}")
    print(f"{'─'*50}")

    champion_path = domain_path / "champion_archetype.json"
    if champion_path.exists():
        try:
            champ = json.loads(champion_path.read_text())
            print(f"Champion: {champ.get('name', 'unknown')}")
            print(f"  {champ.get('philosophy', '')}")
        except Exception:
            pass


if __name__ == "__main__":
    from cli import main
    main()
