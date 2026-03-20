"""
ground.py — Reality grounding layer.

Three capabilities that sanity-check the simulation against domain reality:

1. Anomaly detection:   Pure statistics — no AI call.
   Finds score outliers, dead parameters, uniform-score rounds, batch drift.

2. Calibration audit:   Sonnet analyzes whether distributions are realistic.
   Runs every Nth batch.

3. Deep-dive sampling:  Haiku reviews random rounds for real-world plausibility.

Results feed into hypothesis tracking, simulation evolution, and the thinking log.

Usage:
  from ground import run_grounding, append_grounding_to_thinking_log
  report = run_grounding(domain_path, batch_result, analysis)
"""

import json
import math
import os
import random
import statistics
from datetime import datetime
from pathlib import Path

from api import structured_ai_call
from utils import load_env, load_world_model, load_jsonl

MODEL_CALIBRATION = os.environ.get("AUTOFORGE_GROUND_MODEL", "claude-sonnet-4-6")
MODEL_DEEP_DIVE   = os.environ.get("AUTOFORGE_DEEP_DIVE_MODEL", "claude-haiku-4-5-20251001")


CALIBRATION_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array", "items": {"type": "string"},
        },
        "verdict": {
            "type": "string",
            "enum": ["grounded", "suspect", "miscalibrated"],
        },
        "unrealistic_distributions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parameter":       {"type": "string"},
                    "issue":           {"type": "string"},
                    "suggested_range": {"type": "string"},
                },
                "required": ["parameter", "issue", "suggested_range"],
                "additionalProperties": False,
            },
        },
        "sim_concerns": {
            "type": "array", "items": {"type": "string"},
        },
        "new_hypotheses": {
            "type": "array", "items": {"type": "string"},
        },
    },
    "required": ["findings", "verdict", "unrealistic_distributions", "sim_concerns", "new_hypotheses"],
    "additionalProperties": False,
}

DEEP_DIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "round_reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "round_num":   {"type": "integer"},
                    "plausible":   {"type": "boolean"},
                    "explanation": {"type": "string"},
                },
                "required": ["round_num", "plausible", "explanation"],
                "additionalProperties": False,
            },
        },
        "overall_assessment": {"type": "string"},
        "new_hypotheses": {
            "type": "array", "items": {"type": "string"},
        },
        "sim_concerns": {
            "type": "array", "items": {"type": "string"},
        },
    },
    "required": ["round_reviews", "overall_assessment", "new_hypotheses", "sim_concerns"],
    "additionalProperties": False,
}


def _load_recent_rounds(domain_path: Path, last_n: int = 0) -> list[dict]:
    rounds = load_jsonl(domain_path / "tournament_log.jsonl")
    if last_n > 0:
        return rounds[-last_n:]
    return rounds


def _pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 10:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _log_grounding(domain_path: Path, report: dict):
    log_path = domain_path / "grounding_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(report) + "\n")


def _check_score_outliers(rounds: list[dict]) -> list[dict]:
    scores = [r.get("score", 0) for r in rounds if r.get("score") is not None]
    if len(scores) < 10:
        return []
    avg = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    if stdev < 0.001:
        return []

    anomalies = []
    outlier_rounds = [r for r in rounds if r.get("score", 0) > avg + 2 * stdev]
    if outlier_rounds:
        anomalies.append({
            "type": "score_outlier",
            "severity": "medium" if len(outlier_rounds) < len(rounds) * 0.05 else "high",
            "description": f"{len(outlier_rounds)} rounds scored >2 stdev above mean ({avg:.2f} + 2×{stdev:.2f})",
            "evidence": {
                "count": len(outlier_rounds),
                "example_rounds": [r.get("round") for r in outlier_rounds[:5]],
                "batch_avg": round(avg, 4),
                "batch_stdev": round(stdev, 4),
            },
        })
    return anomalies


def _check_score_uniformity(rounds: list[dict], threshold: float = 0.05) -> list[dict]:
    uniform_count = 0
    for r in rounds:
        contenders = r.get("contenders", [])
        if len(contenders) < 2:
            continue
        scores = [c.get("score", 0) for c in contenders]
        if max(scores) - min(scores) < threshold * max(abs(max(scores)), 1):
            uniform_count += 1

    if uniform_count == 0:
        return []
    pct = uniform_count / len(rounds) * 100
    severity = "high" if pct > 30 else ("medium" if pct > 15 else "low")
    return [{
        "type": "uniform_scores",
        "severity": severity,
        "description": f"{uniform_count}/{len(rounds)} rounds ({pct:.0f}%) had all contenders within {threshold:.0%} of each other",
        "evidence": {"uniform_count": uniform_count, "total_rounds": len(rounds), "pct": round(pct, 1)},
    }]


def _check_dead_parameters(rounds: list[dict]) -> list[dict]:
    param_data: dict[str, tuple[list, list]] = {}
    for r in rounds:
        for c in r.get("contenders", []):
            score = c.get("score")
            strategy = c.get("strategy", {})
            if score is None or not strategy:
                continue
            for key, val in strategy.items():
                if not isinstance(val, (int, float)):
                    continue
                if key not in param_data:
                    param_data[key] = ([], [])
                param_data[key][0].append(val)
                param_data[key][1].append(score)

    anomalies = []
    for param, (vals, scores) in param_data.items():
        r = _pearson(vals, scores)
        if abs(r) < 0.05 and len(vals) >= 30:
            anomalies.append({
                "type": "dead_parameter",
                "severity": "medium",
                "description": f"Parameter '{param}' has near-zero correlation with score (r={r:.3f}, n={len(vals)})",
                "evidence": {"parameter": param, "correlation": round(r, 4), "n": len(vals)},
            })
    return anomalies


def _check_batch_drift(current_rounds: list[dict], prior_rounds: list[dict]) -> list[dict]:
    if not prior_rounds:
        return []
    current_scores = [r.get("score", 0) for r in current_rounds if r.get("score") is not None]
    prior_scores = [r.get("score", 0) for r in prior_rounds if r.get("score") is not None]
    if len(current_scores) < 10 or len(prior_scores) < 10:
        return []

    curr_avg = statistics.mean(current_scores)
    prior_avg = statistics.mean(prior_scores)
    curr_stdev = statistics.stdev(current_scores) if len(current_scores) > 1 else 0
    prior_stdev = statistics.stdev(prior_scores) if len(prior_scores) > 1 else 0

    pooled_stdev = math.sqrt((curr_stdev ** 2 + prior_stdev ** 2) / 2) if (curr_stdev + prior_stdev) > 0 else 1
    if pooled_stdev < 0.001:
        return []

    shift = abs(curr_avg - prior_avg) / pooled_stdev
    if shift < 1.0:
        return []

    direction = "up" if curr_avg > prior_avg else "down"
    severity = "high" if shift > 2.0 else "medium"
    return [{
        "type": "batch_drift",
        "severity": severity,
        "description": f"Score distribution shifted {direction} by {shift:.1f} stdev (prior avg {prior_avg:.2f} → current avg {curr_avg:.2f})",
        "evidence": {
            "prior_avg": round(prior_avg, 4),
            "current_avg": round(curr_avg, 4),
            "shift_stdev": round(shift, 2),
            "direction": direction,
        },
    }]


def detect_anomalies(domain_path: Path, batch_result: dict) -> list[dict]:
    """Scan tournament results for statistical anomalies. Pure Python, no AI calls."""
    n_rounds = batch_result.get("n_rounds", 0)
    if n_rounds == 0:
        return []

    all_rounds = _load_recent_rounds(domain_path)
    if not all_rounds:
        return []

    current = all_rounds[-n_rounds:]
    prior = all_rounds[-2 * n_rounds:-n_rounds] if len(all_rounds) >= 2 * n_rounds else []

    anomalies = []
    anomalies.extend(_check_score_outliers(current))
    anomalies.extend(_check_score_uniformity(current))
    anomalies.extend(_check_dead_parameters(current))
    anomalies.extend(_check_batch_drift(current, prior))

    return anomalies


def calibration_audit(domain_path: Path, batch_result: dict, anomalies: list[dict]) -> dict:
    """Ask Sonnet to analyze whether the simulation's distributions are realistic."""
    load_env(domain_path)

    sim_path = domain_path / "simulation.py"
    sim_source = sim_path.read_text() if sim_path.exists() else "(not found)"
    world_model = load_world_model(domain_path)

    anomaly_text = "\n".join(
        f"- [{a['severity']}] {a['description']}" for a in anomalies
    ) if anomalies else "(none detected)"

    system_prompt = (
        "You are a domain expert auditing whether a simulation is realistic. "
        "Check if the probability distributions, score ranges, and parameter effects "
        "match what would happen in the real world for this domain. "
        "Be specific — name exact parameters, values, and what the real-world range should be."
    )

    user_prompt = f"""WORLD MODEL:
{world_model}

SIMULATION SOURCE CODE:
```python
{sim_source}
```

BATCH RESULTS:
  Rounds: {batch_result['n_rounds']}
  Avg score: {batch_result['avg_score']}
  Best score: {batch_result['best_score']}
  Worst score: {batch_result['worst_score']}
  Score last 10: {batch_result.get('score_last_10', [])}

DETECTED ANOMALIES:
{anomaly_text}

Audit this simulation for realism:
1. Are the probability distributions in random_state() realistic?
2. Does simulate() compute expected value correctly?
3. Are the score ranges reasonable for this domain?
4. Do all CANDIDATE_SCHEMA parameters meaningfully affect the outcome?

Verdict:
  grounded      = distributions and scoring logic are realistic
  suspect       = some distributions seem off but the structure is sound
  miscalibrated = fundamental issues with how the simulation models reality
"""

    return structured_ai_call(
        task_name="ground_calibration",
        domain_path=domain_path,
        model=MODEL_CALIBRATION,
        max_tokens=4096,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=CALIBRATION_SCHEMA,
    )


def deep_dive_sample(domain_path: Path, n: int = 5) -> dict:
    """Pull N random rounds from tournament_log.jsonl for domain-expert sanity checking."""
    load_env(domain_path)

    all_rounds = _load_recent_rounds(domain_path)
    with_contenders = [r for r in all_rounds if r.get("contenders")]
    pool = with_contenders if with_contenders else all_rounds
    if not pool:
        return {
            "round_reviews": [],
            "overall_assessment": "No rounds available for review.",
            "new_hypotheses": [],
            "sim_concerns": [],
        }

    sampled = random.sample(pool, min(n, len(pool)))
    world_model = load_world_model(domain_path)

    rounds_text = ""
    for r in sampled:
        contender_lines = ""
        for c in r.get("contenders", [])[:4]:
            contender_lines += f"    {c.get('name', '?')}: score={c.get('score', '?')} strategy={json.dumps(c.get('strategy', {}))}\n"

        rounds_text += f"""
--- Round {r.get('round', '?')} ---
State: {json.dumps(r.get('state', {}), indent=2)}
Winner: {r.get('archetype', '?')} (score={r.get('score', '?')})
Winner strategy: {json.dumps(r.get('winner', {}))}
Score margin: {r.get('score_margin', '?')}
Contenders:
{contender_lines}
"""

    system_prompt = (
        "You are a domain expert reviewing individual simulation results. "
        "For each round, assess: would this outcome happen in the real world? "
        "Does the winning strategy make sense given the scenario?"
    )

    user_prompt = f"""WORLD MODEL:
{world_model}

SAMPLED ROUNDS FOR REVIEW:
{rounds_text}

For each round:
1. Is the winner's score plausible given the state/scenario?
2. Does the winning strategy make domain sense for this scenario?
3. Would the margin between winner and losers be realistic?

Flag any round where the simulation's output doesn't match real-world expectations.
Generate hypotheses about domain patterns you notice across rounds.
"""

    return structured_ai_call(
        task_name="ground_deep_dive",
        domain_path=domain_path,
        model=MODEL_DEEP_DIVE,
        max_tokens=2048,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=DEEP_DIVE_SCHEMA,
    )


def run_grounding(
    domain_path: Path,
    batch_result: dict,
    analysis: dict,
    *,
    sample_n: int = 5,
    skip_calibration_audit: bool = False,
) -> dict:
    """Run the full grounding pipeline for one batch."""
    report = {
        "anomalies": [],
        "calibration_findings": [],
        "calibration_verdict": "skipped",
        "deep_dive_findings": [],
        "new_hypotheses": [],
        "sim_concerns": [],
        "timestamp": datetime.now().isoformat(),
    }

    try:
        report["anomalies"] = detect_anomalies(domain_path, batch_result)
    except Exception as e:
        report["anomalies"] = [{"type": "error", "severity": "low", "description": f"Anomaly detection failed: {e}", "evidence": {}}]

    if not skip_calibration_audit:
        try:
            cal = calibration_audit(domain_path, batch_result, report["anomalies"])
            report["calibration_findings"] = cal.get("findings", [])
            report["calibration_verdict"] = cal.get("verdict", "skipped")
            report["sim_concerns"].extend(cal.get("sim_concerns", []))
            report["new_hypotheses"].extend(cal.get("new_hypotheses", []))
            for ud in cal.get("unrealistic_distributions", []):
                report["sim_concerns"].append(
                    f"{ud['parameter']}: {ud['issue']} (suggested: {ud['suggested_range']})"
                )
        except Exception as e:
            report["calibration_findings"] = [f"Calibration audit failed: {e}"]

    try:
        dive = deep_dive_sample(domain_path, n=sample_n)
        report["deep_dive_findings"] = [
            f"Round {r['round_num']}: {'plausible' if r['plausible'] else 'IMPLAUSIBLE'} — {r['explanation']}"
            for r in dive.get("round_reviews", [])
        ]
        report["new_hypotheses"].extend(dive.get("new_hypotheses", []))
        report["sim_concerns"].extend(dive.get("sim_concerns", []))
    except Exception as e:
        report["deep_dive_findings"] = [f"Deep-dive sampling failed: {e}"]

    report["new_hypotheses"] = list(dict.fromkeys(report["new_hypotheses"]))
    report["sim_concerns"] = list(dict.fromkeys(report["sim_concerns"]))

    _log_grounding(domain_path, report)
    return report


def append_grounding_to_thinking_log(log_path: Path, batch_num: int, report: dict):
    """Append grounding results to the thinking log markdown."""
    anomalies = report.get("anomalies", [])
    cal_verdict = report.get("calibration_verdict", "skipped")
    cal_findings = report.get("calibration_findings", [])
    dive_findings = report.get("deep_dive_findings", [])
    new_hyp = report.get("new_hypotheses", [])
    sim_concerns = report.get("sim_concerns", [])

    anomaly_lines = chr(10).join(f"- [{a.get('severity', '?')}] {a['description']}" for a in anomalies) if anomalies else "- none"
    finding_lines = chr(10).join(f"- {f}" for f in cal_findings) if cal_findings else "- (skipped)"
    dive_lines = chr(10).join(f"- {f}" for f in dive_findings) if dive_findings else "- none"
    hyp_lines = chr(10).join(f"- {h}" for h in new_hyp) if new_hyp else "- none"
    concern_lines = chr(10).join(f"- {c}" for c in sim_concerns) if sim_concerns else "- none"

    section = f"""
### Grounding — Batch {batch_num}

**Anomalies detected:** {len(anomalies)}
{anomaly_lines}

**Calibration verdict:** {cal_verdict}
{finding_lines}

**Deep-dive review:**
{dive_lines}

**Simulation concerns:**
{concern_lines}

**New hypotheses from grounding:**
{hyp_lines}

"""
    with open(log_path, "a") as f:
        f.write(section)
