"""
validate.py — Simulation health: validate, calibrate, generate evals.
"""

import json
import os
import sys
import importlib
from pathlib import Path

from api import structured_ai_call
from utils import ENGINE_ROOT, DOMAINS_ROOT, load_env, load_sim, load_world_model, random_candidate_from_schema, midpoint_candidate_from_schema


# ── Eval generation ───────────────────────────────────────────────────────────

MODEL_EVALS = os.environ.get("AUTOFORGE_EVALS_MODEL", "claude-opus-4-6")

_EVAL_GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id":          {"type": "string"},
                    "description": {"type": "string"},
                    "state":       {"type": "string"},
                    "min_score":   {"type": "number"},
                },
                "required": ["id", "description", "state", "min_score"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["scenarios", "rationale"],
    "additionalProperties": False,
}


def generate_evals(domain_path: Path, n: int = 10) -> list:
    """
    Read world_model.md + simulation.py, generate eval scenarios.
    Writes to evals/scenarios.jsonl. Returns list of valid scenario dicts.
    """
    load_env(domain_path)
    world_model = load_world_model(domain_path)
    if not world_model:
        print("  No world_model.md found — cannot generate evals")
        return []

    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    try:
        sim = importlib.import_module("simulation")
        sample_states = [sim.random_state() for _ in range(5)]
        metric_name = sim.METRIC_NAME
    except Exception as e:
        print(f"  Cannot import simulation.py: {e}")
        return []

    system_prompt = (
        "You are generating eval scenarios for an autonomous strategy learning system. "
        "Each scenario tests a specific case from the world model — edge cases, "
        "failure modes, abstention boundaries, and critical domain situations. "
        "The state dict must match the exact schema shown in the sample states. "
        "Return each state as a JSON-encoded string in the 'state' field."
    )
    user_prompt = f"""WORLD MODEL:
{world_model}

SAMPLE STATES (your output states must match this schema exactly):
{json.dumps(sample_states, indent=2)}

METRIC: {metric_name}

Generate {n} eval scenarios that test the cases the world model says matter:
- Edge cases where strategies commonly fail
- Scenarios where the specialist should abstain (high uncertainty)
- Failure mode scenarios (what must never happen)
- A mix of easy baseline scenarios and hard edge cases

Set min_score to a reasonable threshold — the minimum acceptable score for
a competent specialist on each scenario.

Return each state as a valid JSON string matching the sample schema above.
"""

    try:
        data = structured_ai_call(
            task_name="generate_evals",
            domain_path=domain_path,
            model=MODEL_EVALS,
            max_tokens=8000,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=_EVAL_GEN_SCHEMA,
            thinking=True,
        )
    except Exception as e:
        print(f"  Eval generation failed: {e}")
        return []

    expected_keys = set(sample_states[0].keys()) if sample_states else set()
    valid = []
    for scenario in data.get("scenarios", []):
        try:
            state = json.loads(scenario["state"]) if isinstance(scenario["state"], str) else scenario["state"]
            if not isinstance(state, dict):
                continue
            if expected_keys and not (set(state.keys()) >= expected_keys):
                continue
            valid.append({
                "id":          scenario["id"],
                "description": scenario["description"],
                "state":       state,
                "min_score":   scenario["min_score"],
            })
        except (json.JSONDecodeError, KeyError):
            continue

    if not valid:
        print("  No valid scenarios generated")
        return []

    evals_dir = domain_path / "evals"
    evals_dir.mkdir(exist_ok=True)
    with open(evals_dir / "scenarios.jsonl", "w") as f:
        for s in valid:
            f.write(json.dumps(s) + "\n")

    return valid


# ── Simulation validation ─────────────────────────────────────────────────────

def _print_validation(ok: list, warnings: list, errors: list):
    for msg in ok:
        print(f"  [ok] {msg}")
    for msg in warnings:
        print(f"  [warn] {msg}")
    for msg in errors:
        print(f"  [error] {msg}")


def validate_domain(domain_path) -> tuple:
    """
    Validate a domain's simulation.py. Returns (ok, warnings, errors) lists.
    Safe to call from other commands — does not sys.exit.
    """
    import collections as _col
    domain_path = Path(domain_path)

    errors   = []
    warnings = []
    ok       = []

    try:
        sim = load_sim(domain_path)
        ok.append("simulation.py imports cleanly")
    except ImportError as e:
        errors.append(str(e))
        return ok, warnings, errors

    for name in ["simulate", "random_state", "CANDIDATE_SCHEMA", "METRIC_NAME"]:
        if hasattr(sim, name):
            ok.append(f"{name} exists")
        else:
            errors.append(f"Missing required export: {name}")

    if errors:
        return ok, warnings, errors

    if isinstance(sim.METRIC_NAME, str) and sim.METRIC_NAME.strip():
        ok.append(f"METRIC_NAME = '{sim.METRIC_NAME}'")
    else:
        errors.append("METRIC_NAME must be a non-empty string")

    schema = sim.CANDIDATE_SCHEMA
    if isinstance(schema, dict) and "properties" in schema and schema["properties"]:
        n_params = len(schema["properties"])
        ok.append(f"CANDIDATE_SCHEMA has {n_params} parameter(s): {', '.join(schema['properties'].keys())}")
    else:
        errors.append("CANDIDATE_SCHEMA must be a dict with a non-empty 'properties' key")

    if errors:
        return ok, warnings, errors

    try:
        state = sim.random_state()
        if isinstance(state, dict) and state:
            ok.append(f"random_state() returns dict with {len(state)} key(s): {', '.join(list(state.keys())[:5])}")
        else:
            errors.append("random_state() must return a non-empty dict")
    except Exception as e:
        errors.append(f"random_state() raised: {e}")

    try:
        candidate = midpoint_candidate_from_schema(schema)

        state = sim.random_state()
        score = sim.simulate(candidate, state)

        if isinstance(score, (int, float)):
            ok.append(f"simulate() returns {sim.METRIC_NAME} = {score}")
        else:
            errors.append(f"simulate() must return a number, got {type(score).__name__}")
    except Exception as e:
        errors.append(f"simulate() raised: {e}")

    try:
        scores = [sim.simulate(candidate, sim.random_state()) for _ in range(20)]
        unique = len(set(round(s, 4) for s in scores))
        if unique < 5:
            warnings.append(f"simulate() returned only {unique} unique scores across 20 random scenarios — check calibration")
        else:
            ok.append(f"simulate() produces varied scores across scenarios ({unique}/20 unique)")
    except Exception:
        pass

    try:
        _candidates = [random_candidate_from_schema(schema) for _ in range(4)]
        win_counts = _col.Counter()
        for _ in range(50):
            state = sim.random_state()
            winner_idx = max(range(len(_candidates)), key=lambda j: sim.simulate(_candidates[j], state))
            win_counts[winner_idx] += 1
        top_pct = max(win_counts.values()) / 50
        if top_pct > 0.8:
            warnings.append(f"One random candidate wins {top_pct:.0%} of 50 rounds — possible dominant strategy")
        else:
            ok.append(f"No single candidate dominates (top win rate {top_pct:.0%} across 4 candidates)")
    except Exception:
        pass

    if hasattr(sim, "build_context"):
        try:
            ctxs = [sim.build_context(sim.random_state()) for _ in range(30)]
            if ctxs and isinstance(ctxs[0], dict):
                categorical_keys = [k for k, v in ctxs[0].items() if isinstance(v, str)]
                diverse = sum(1 for k in categorical_keys if len(set(c.get(k) for c in ctxs)) >= 2)
                if categorical_keys and diverse == 0:
                    warnings.append("build_context() returns same categorical values every time")
                else:
                    ok.append(f"build_context() present, {len(ctxs[0])} key(s), categorically diverse")
            else:
                warnings.append("build_context() returned non-dict — expected dict")
        except Exception as e:
            warnings.append(f"build_context() raised: {e}")

    wm_path = domain_path / "world_model.md"
    if wm_path.exists():
        ok.append("world_model.md exists")
    else:
        has_legacy = False
        for fname in ["brain.md", "extract.md", "director.md"]:
            p = domain_path / "prompts" / fname
            if p.exists():
                ok.append(f"prompts/{fname} exists (legacy)")
                has_legacy = True
            else:
                warnings.append(f"prompts/{fname} missing — needed for AI archetype generation")
        if has_legacy:
            warnings.append("Using legacy prompt files — consider migrating to world_model.md")

    return ok, warnings, errors


def cmd_validate(args):
    """Sanity-check simulation.py: imports, required exports, schema, sim output."""
    domain_path = DOMAINS_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    ok, warnings, errors = validate_domain(domain_path)
    _print_validation(ok, warnings, errors)

    if errors:
        sys.exit(1)
    else:
        print(f"\nDomain {args.domain} looks good. Run:")
        print(f"  autoforge run --domain {args.domain} --batches 3 --rounds 50")


def cmd_calibrate(args):
    """Show scenario distributions, score stats, and dominance check."""
    import statistics
    import collections

    domain_path = DOMAINS_ROOT / args.domain
    if not domain_path.exists():
        print(f"Domain not found: {domain_path}")
        sys.exit(1)

    try:
        sim = load_sim(domain_path)
    except ImportError as e:
        print(e)
        sys.exit(1)

    n = args.n
    print(f"\nCalibrating {args.domain} — sampling {n} scenarios...\n")

    states = [sim.random_state() for _ in range(n)]

    schema = sim.CANDIDATE_SCHEMA
    schema_props = schema.get("properties", {})

    def random_candidate():
        return random_candidate_from_schema(schema)

    def degenerate_candidates():
        defs = []
        for label, pick in [("min-all", "min"), ("max-all", "max"), ("mid-all", "mid")]:
            c = {}
            for key, spec in schema_props.items():
                if spec.get("type") == "number":
                    lo, hi = spec.get("minimum", 0.0), spec.get("maximum", 1.0)
                    c[key] = lo if pick == "min" else (hi if pick == "max" else (lo + hi) / 2)
                elif spec.get("type") == "integer":
                    lo, hi = spec.get("minimum", 0), spec.get("maximum", 10)
                    c[key] = lo if pick == "min" else (hi if pick == "max" else (lo + hi) // 2)
                elif "enum" in spec:
                    vals = spec["enum"]
                    c[key] = vals[0] if pick in ("min", "mid") else vals[-1]
                else:
                    c[key] = None
            defs.append((label, c))
        return defs

    adversaries   = degenerate_candidates()
    candidates    = [random_candidate() for _ in range(8)]
    all_candidates = candidates + [c for _, c in adversaries]

    win_counts    = collections.Counter()
    adversary_wins = collections.Counter()
    all_scores    = []

    for state in states:
        scored = [(sim.simulate(c, state), i) for i, c in enumerate(all_candidates)]
        scored.sort(reverse=True)
        all_scores.append(scored[0][0])
        winner_idx = scored[0][1]
        win_counts[winner_idx] += 1
        if winner_idx >= 8:
            adversary_wins[adversaries[winner_idx - 8][0]] += 1

    print(f"\nScenario Distribution ({n} samples)")
    print(f"  {'Key':<24} {'Type':<12} Distribution")
    print(f"  {'─'*72}")
    for k in list(states[0].keys()):
        vals = [s[k] for s in states if k in s]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            true_pct = sum(1 for v in vals if v) / len(vals) * 100
            dist = f"{true_pct:.0f}% True  /  {100-true_pct:.0f}% False"
            typ = "bool"
        elif isinstance(vals[0], (int, float)):
            dist = f"min {min(vals):.2f}  median {statistics.median(vals):.2f}  max {max(vals):.2f}"
            typ = "number"
        else:
            counts = collections.Counter(vals)
            top = sorted(counts.items(), key=lambda x: -x[1])[:6]
            dist = "  ".join(f"{v}: {c/len(vals)*100:.0f}%" for v, c in top)
            typ = "categorical"
        print(f"  {k:<24} {typ:<12} {dist}")

    s   = sorted(all_scores)
    q1  = s[len(s) // 4]
    q3  = s[3 * len(s) // 4]
    print(f"\nScore Distribution ({sim.METRIC_NAME})")
    for label, val in [("min", min(all_scores)), ("p25", q1),
                       ("median", statistics.median(all_scores)),
                       ("p75", q3), ("max", max(all_scores)),
                       ("stdev", statistics.stdev(all_scores))]:
        print(f"  {label:<8} {val:.2f}")

    print(f"\nCandidate Win Distribution (8 random strategies)")
    print(f"  {'Candidate':<14} {'Wins':>6} {'Win %':>7}")
    print(f"  {'─'*30}")
    for i in range(len(candidates)):
        wins = win_counts.get(i, 0)
        print(f"  {'candidate_'+str(i):<14} {wins:>6} {wins/n*100:>6.0f}%")

    print(f"\nSanity Adversary Check (degenerate candidates — should lose)")
    print(f"  {'Adversary':<12} {'Wins':>6} {'Win %':>7}  Status")
    print(f"  {'─'*38}")
    for label, _ in adversaries:
        wins = adversary_wins.get(label, 0)
        pct  = wins / n * 100
        status = "FAIL" if pct > 25 else ("warn" if pct > 10 else "ok")
        print(f"  {label:<12} {wins:>6} {pct:>6.0f}%  {status}")

    issues = []
    if max(all_scores) == min(all_scores):
        issues.append("All scores identical — simulate() may not depend on scenario state")
    elif statistics.stdev(all_scores) < abs(statistics.mean(all_scores)) * 0.05:
        issues.append("Very low score variance — check that scenario factors affect outcomes")
    random_wins_only = {k: v for k, v in win_counts.items() if k < 8}
    if random_wins_only:
        top_pct = max(random_wins_only.values()) / n * 100
        if top_pct > 70:
            issues.append(f"One random candidate wins {top_pct:.0f}% of rounds — likely a dominant strategy")
    for label, _ in adversaries:
        adv_pct = adversary_wins.get(label, 0) / n * 100
        if adv_pct > 25:
            issues.append(f"Sanity adversary '{label}' wins {adv_pct:.0f}% of rounds — sim may reward degenerate strategies")
    zero_pct = sum(1 for sc in all_scores if sc == 0) / len(all_scores) * 100
    if zero_pct > 30:
        issues.append(f"{zero_pct:.0f}% of rounds score 0 — check simulate() edge cases")

    if issues:
        print("\nCalibration warnings:")
        for issue in issues:
            print(f"  ! {issue}")
        print(f"\n  Edit {args.domain}/simulation.py and re-run calibrate.")
    else:
        print(f"\nCalibration looks good.")
        print(f"  autoforge run --domain {args.domain} --batches 3 --rounds 50")
