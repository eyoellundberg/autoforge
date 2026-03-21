"""
export.py — Stage 3 training data exporter.

Reads tournament_log.jsonl and exports:
  - training_data.jsonl      instruction-tuning JSONL (MLX-LM / OpenAI format)
  - training_preferences.jsonl  pairwise preference data
  - training_features.csv    flat feature matrix for XGBoost (numerical domains)
"""

import json
import csv
import statistics
from pathlib import Path

from utils import load_mission, load_world_model, normalize_confidence, load_jsonl


_NON_SCORE_KEYS = {"round", "winner", "state", "archetype", "metric", "score_margin", "contenders"}


def _detect_score(entry: dict) -> float:
    if "score" in entry:
        return float(entry["score"])
    for key, value in entry.items():
        if key not in _NON_SCORE_KEYS:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    raise KeyError(f"Cannot find score in log entry: {list(entry.keys())}")


def _format_scenario(state: dict) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in state.items())


def _format_playbook_context(playbook: list) -> str:
    if not playbook:
        return "(no principles yet)"
    lines = []
    for p in playbook:
        conf = normalize_confidence(p.get("confidence", 0))
        ctx  = p.get("context", "")
        text = p.get("principle", "")
        lines.append(f"- [{conf:.0%}] [{ctx}] {text}")
    return "\n".join(lines)


def _format_strategy(strategy: dict) -> str:
    return json.dumps(strategy, sort_keys=True)


def _verbalize_response(state: dict, winner: dict, score: float, playbook: list) -> str:
    """
    Build a natural language recommendation from tournament winner data.
    Uses playbook principles as the reasoning chain.
    """
    # Find the most relevant principles (match on context keys present in state)
    state_keys = set(str(v).lower() for v in state.values())
    relevant = []
    for p in playbook:
        ctx = p.get("context", "").lower()
        principle = p.get("principle", "")
        condition = p.get("condition") or ""
        if not principle:
            continue
        # Loosely match principles whose context overlaps with current state
        if ctx and any(k in ctx for k in state_keys):
            relevant.append((p.get("confidence", 0), principle, condition))
        else:
            relevant.append((p.get("confidence", 0) * 0.5, principle, condition))

    relevant.sort(reverse=True)
    top_principles = relevant[:3]

    # Format state context
    state_summary = "  ".join(f"{k}: {v}" for k, v in list(state.items())[:6])

    # Format strategy parameters cleanly
    strategy_lines = "\n".join(f"  {k}: {v}" for k, v in winner.items())

    # Build response
    lines = [f"Scenario: {state_summary}", ""]
    lines.append("Recommended strategy:")
    lines.append(strategy_lines)
    lines.append("")

    if top_principles:
        lines.append("Reasoning:")
        for _, principle, condition in top_principles:
            if condition:
                lines.append(f"  - {principle} (when {condition})")
            else:
                lines.append(f"  - {principle}")
        lines.append("")

    lines.append(f"Expected outcome: {score:.3f}")
    return "\n".join(lines)


def _detect_domain_type(domain_path: Path) -> str:
    """
    Returns "numerical" if all CANDIDATE_SCHEMA params are numbers/integers/enums,
    "language" if any params are free-form strings.
    """
    import sys as _sys, importlib
    _sys.path.insert(0, str(domain_path))
    try:
        sim = importlib.import_module("simulation")
        schema = getattr(sim, "CANDIDATE_SCHEMA", {})
        props  = schema.get("properties", {})
        for spec in props.values():
            t = spec.get("type")
            if t == "string" and "enum" not in spec:
                return "language"
        return "numerical"
    except Exception:
        return "numerical"
    finally:
        _sys.path.pop(0)


def export_training_data(domain_path: Path):
    """
    Export tournament_log.jsonl as instruction-tuning JSONL.
    Quality filter: only rounds where score >= median.
    For numerical domains, also exports training_features.csv.
    """
    log_path = domain_path / "tournament_log.jsonl"
    pb_path  = domain_path / "playbook.jsonl"
    out_path = domain_path / "training_data.jsonl"

    if not log_path.exists():
        print(f"No tournament_log.jsonl found in {domain_path}")
        print("Run some batches first before exporting.")
        return

    entries = load_jsonl(log_path)
    if not entries:
        print("tournament_log.jsonl is empty — nothing to export.")
        return

    total_rounds = len(entries)

    scored = []
    for entry in entries:
        try:
            score = _detect_score(entry)
            scored.append((score, entry))
        except (KeyError, ValueError):
            continue

    if not scored:
        print("Could not find score values in tournament_log.jsonl — check log format.")
        return

    all_scores = [s for s, _ in scored]
    median_score = statistics.median(all_scores)

    kept = [(s, e) for s, e in scored if s >= median_score]
    print(f"Total rounds: {total_rounds}")
    print(f"After quality filter (score >= median {median_score:.2f}): {len(kept)} kept")

    domain_type = _detect_domain_type(domain_path)

    playbook = []
    if pb_path.exists():
        playbook = [json.loads(l) for l in pb_path.read_text().splitlines() if l.strip()]
    playbook_context = _format_playbook_context(playbook)

    world_model = load_world_model(domain_path)

    system_parts = ["You are a strategy expert for this domain."]
    if world_model:
        system_parts.append(f"World Model:\n{world_model}")
    else:
        mission_text = load_mission(domain_path)
        if mission_text:
            system_parts.append(f"Mission:\n{mission_text}")
    system_parts.append(f"Playbook principles:\n{playbook_context}")
    system_content = "\n\n".join(system_parts)

    # Identify breakthrough batches — examples from these get written twice
    bt_path = domain_path / "breakthroughs.jsonl"
    breakthrough_batches = set()
    if bt_path.exists():
        for line in bt_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    breakthrough_batches.add(json.loads(line)["batch"])
                except (KeyError, json.JSONDecodeError):
                    pass
    if breakthrough_batches:
        print(f"Breakthrough batches (2x weight): {sorted(breakthrough_batches)}")

    written = 0
    with open(out_path, "w") as f:
        for score, entry in kept:
            state  = entry.get("state", {})
            winner = entry.get("winner", {})
            if not state or not winner:
                continue
            example = {
                "messages": [
                    {"role": "system",    "content": system_content},
                    {"role": "user",      "content": f"Scenario:\n{_format_scenario(state)}"},
                    {"role": "assistant", "content": _verbalize_response(state, winner, score, playbook)},
                ]
            }
            f.write(json.dumps(example) + "\n")
            written += 1
            if entry.get("batch") in breakthrough_batches:
                f.write(json.dumps(example) + "\n")
                written += 1

    pref_path = domain_path / "training_preferences.jsonl"
    pref_written = 0
    with open(pref_path, "w") as f:
        for score, entry in kept:
            state      = entry.get("state", {})
            winner     = entry.get("winner", {})
            contenders = entry.get("contenders", [])
            if not state or not winner or len(contenders) < 2:
                continue

            prompt = (
                f"{system_content}\n\n"
                f"Scenario:\n{_format_scenario(state)}\n\n"
                "Choose the stronger strategy for this scenario."
            )
            winner_text = _format_strategy(winner)
            for contender in contenders[1:]:
                loser = contender.get("strategy", {})
                if not loser:
                    continue
                preference = {
                    "prompt": prompt,
                    "chosen": winner_text,
                    "rejected": _format_strategy(loser),
                    "metadata": {
                        "winner_name": entry.get("archetype", ""),
                        "rejected_name": contender.get("name", ""),
                        "winner_score": score,
                        "rejected_score": contender.get("score"),
                        "score_margin": round(score - float(contender.get("score", 0.0)), 4),
                    },
                }
                f.write(json.dumps(preference) + "\n")
                pref_written += 1

    if domain_type == "numerical":
        state_keys  = []
        winner_keys = []
        for _, entry in kept:
            state  = entry.get("state", {})
            winner = entry.get("winner", {})
            for k in state:
                if k not in state_keys:
                    state_keys.append(k)
            for k in winner:
                if k not in winner_keys:
                    winner_keys.append(k)

        all_margins = [e.get("score_margin", None) for _, e in kept]
        has_margins = any(m is not None for m in all_margins)
        if has_margins:
            margin_values = sorted(m for m in all_margins if m is not None)
            p25_idx = max(0, int(len(margin_values) * 0.25) - 1)
            p25_margin = margin_values[p25_idx] if margin_values else 0.0
        else:
            p25_margin = None

        csv_path = domain_path / "training_features.csv"
        extra_cols = ["score", "score_margin", "uncertain"] if has_margins else ["score"]
        fieldnames = state_keys + winner_keys + extra_cols
        n_cols = len(fieldnames)

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            csv_rows = 0
            for score, entry in kept:
                state  = entry.get("state", {})
                winner = entry.get("winner", {})
                if not state or not winner:
                    continue
                row = {}
                for k in state_keys:
                    row[k] = state.get(k, "")
                for k in winner_keys:
                    row[k] = winner.get(k, "")
                row["score"] = score
                if has_margins:
                    margin = entry.get("score_margin", 0.0)
                    row["score_margin"] = margin
                    row["uncertain"] = 1 if margin < p25_margin else 0
                writer.writerow(row)
                csv_rows += 1

        domain_name = domain_path.name
        print(f"\nDomain type: numerical")
        print(f"  training_features.csv  — {csv_rows} rows, {n_cols} columns")
        print(f"  training_data.jsonl    — {written} examples (MLX-LM fine-tuning)")
        if pref_written:
            print(f"  training_preferences.jsonl — {pref_written} pairwise preferences")
        print(f"""
This is a reward model dataset: features = state + strategy, label = score.
Inference: generate N candidate strategies, score each with XGBoost, pick the best.

  import pandas as pd, xgboost as xgb, numpy as np
  df = pd.read_csv('{domain_name}/training_features.csv')
  feature_cols = [c for c in df.columns if c != 'score']
  X, y = df[feature_cols].values, df['score'].values
  model = xgb.train({{'max_depth': 4, 'eta': 0.1, 'objective': 'reg:squarederror'}},
                    xgb.DMatrix(X, label=y), num_boost_round=200)""")

        if has_margins and p25_margin is not None:
            threshold_path = domain_path / "abstain_threshold.json"
            threshold_payload = {
                "strategy": "score_margin_p25",
                "threshold": round(p25_margin, 6),
                "note": "Escalate when the top candidate's predicted edge is smaller than this threshold.",
            }
            threshold_path.write_text(json.dumps(threshold_payload, indent=2) + "\n")
            print(f"\nAbstain threshold: {round(p25_margin, 3)}  (written to abstain_threshold.json)")

    else:
        domain_name = domain_path.name
        print(f"\nDomain type: language (free-form string params detected)")
        print(f"  training_data.jsonl — {written} examples")
        if pref_written:
            print(f"  training_preferences.jsonl — {pref_written} pairwise preferences")
        print(f"  Fine-tune: mlx_lm.lora --model mlx-community/Qwen3-4B-4bit --data {domain_name}/ --train")
