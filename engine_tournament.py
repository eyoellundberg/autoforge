"""
engine_tournament.py — Central tournament engine for all Autoforge domains.

Domains expose:
    simulate(candidate, state) -> float       required
    random_state() -> dict                    required
    CANDIDATE_SCHEMA: dict                    required
    METRIC_NAME: str                          required
    build_context(state) -> dict              optional  (defaults to full state dict)
    prepare_state(state) -> dict              optional  (expensive per-scenario prep, called once)
    is_event(state) -> bool                   optional  (event/nonevent win split)

Usage:
    from engine_tournament import run_batch
    result = run_batch(domain_path, n_rounds=200, use_brain=True, workers=4)
"""

import heapq
import importlib
import json
import os
import sys
from pathlib import Path

ENGINE_ROOT = Path(__file__).parent

# Module-level simulation reference — set by run_batch (single process)
# and by _init_worker (each worker process in parallel mode)
_SIM = None


def _init_worker(domain_path_str: str):
    """Initialize a worker process with the domain's simulation module."""
    global _SIM
    if domain_path_str not in sys.path:
        sys.path.insert(0, domain_path_str)
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    _SIM = importlib.import_module("simulation")


def _score_round(args):
    """Score all candidates against one state. Module-level for multiprocessing pickling."""
    state, candidates, candidate_names = args
    prepare_state = getattr(_SIM, "prepare_state", None)
    if prepare_state is not None:
        state = prepare_state(state)
    all_results = [(_SIM.simulate(c, state), c, n) for c, n in zip(candidates, candidate_names)]
    return heapq.nlargest(4, all_results, key=lambda x: x[0])


# ── Playbook helpers ──────────────────────────────────────────────────────────

def _load_playbook(domain_path: Path) -> list:
    p = domain_path / "playbook.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _save_playbook(domain_path: Path, playbook: list):
    with open(domain_path / "playbook.jsonl", "w") as f:
        for entry in playbook:
            f.write(json.dumps(entry) + "\n")


# ── Stage 1: evolutionary candidate generation ────────────────────────────────

def _generate_procedural_candidates(domain_path: Path, n: int = 16) -> list:
    """
    Evolutionary Stage 1 candidate generation. No API calls.

    Reads top_candidates.json from the prior batch and evolves from them:
    - Elitism: keep top 2 winners unchanged
    - Mutation: gaussian noise on numeric params, occasional enum flip
    - Crossover: mix parameters from two winners
    - Exploration: random candidates to prevent premature convergence

    First batch (no prior winners): purely random.
    """
    import random

    schema_props = _SIM.CANDIDATE_SCHEMA.get("properties", {})

    def random_candidate() -> dict:
        c = {}
        for key, spec in schema_props.items():
            if spec.get("type") == "number":
                lo = spec.get("minimum", 0.0)
                hi = spec.get("maximum", 1.0)
                c[key] = round(random.uniform(lo, hi), 4)
            elif spec.get("type") == "integer":
                lo = spec.get("minimum", 0)
                hi = spec.get("maximum", 10)
                c[key] = random.randint(lo, hi)
            elif "enum" in spec:
                c[key] = random.choice(spec["enum"])
            else:
                c[key] = None
        return c

    def mutate(base: dict) -> dict:
        child = dict(base)
        for key, spec in schema_props.items():
            if random.random() < 0.3:
                if spec.get("type") == "number":
                    lo = spec.get("minimum", 0.0)
                    hi = spec.get("maximum", 1.0)
                    noise = (hi - lo) * 0.15 * random.gauss(0, 1)
                    child[key] = round(max(lo, min(hi, base[key] + noise)), 4)
                elif spec.get("type") == "integer":
                    lo = spec.get("minimum", 0)
                    hi = spec.get("maximum", 10)
                    delta = random.choice([-2, -1, 1, 2])
                    child[key] = max(lo, min(hi, base[key] + delta))
                elif "enum" in spec:
                    child[key] = random.choice(spec["enum"])
        return child

    def crossover(a: dict, b: dict) -> dict:
        return {k: (a[k] if random.random() < 0.5 else b[k]) for k in schema_props}

    prior = []
    top_path = domain_path / "top_candidates.json"
    if top_path.exists():
        try:
            prior = [e["strategy"] for e in json.loads(top_path.read_text())]
        except Exception:
            pass

    if not prior:
        return [random_candidate() for _ in range(n)]

    candidates = []
    candidates.extend(prior[:2])
    for p in prior[:3]:
        candidates.extend([mutate(p) for _ in range(2)])
    if len(prior) >= 2:
        candidates.extend([crossover(prior[0], prior[i % len(prior)]) for i in range(1, 3)])
    while len(candidates) < n:
        candidates.append(random_candidate())
    return candidates[:n]


# ── Main batch runner ─────────────────────────────────────────────────────────

def run_batch(
    domain_path,
    n_rounds: int,
    generation_offset: int = 0,
    hints: list = None,
    use_brain: bool = False,
    workers: int = 1,
) -> dict:
    """
    Run one batch of the tournament.

    domain_path:       path to the domain folder
    n_rounds:          how many rounds to run
    generation_offset: global round counter offset (for logging)
    hints:             director hints from the prior batch (Stage 2)
    use_brain:         True → Stage 2 (Sonnet generates archetype library)
    workers:           parallel simulation workers

    Returns a result dict consumed by the director call.
    """
    global _SIM

    hints = hints or []
    domain_path = Path(domain_path)
    os.chdir(domain_path)

    # Load the domain's simulation module fresh
    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    _SIM = importlib.import_module("simulation")

    # Optional domain hooks
    build_context = getattr(_SIM, "build_context", None) or (lambda s: dict(s))
    is_event      = getattr(_SIM, "is_event", None)

    playbook  = _load_playbook(domain_path)
    log_path  = domain_path / "tournament_log.jsonl"

    # ── Stage 2: generate archetype library via Sonnet ────────────────────────
    archetypes = []
    if use_brain:
        from engine_brain import call_library
        archetypes = call_library(
            playbook=playbook,
            hints=hints,
            domain_path=domain_path,
            candidate_schema=_SIM.CANDIDATE_SCHEMA,
        )

    # ── Stage 1: generate procedural candidates once per batch ───────────────
    stage1_candidates = []
    stage1_names      = []
    if not archetypes:
        stage1_candidates = _generate_procedural_candidates(domain_path, n=16)
        stage1_names      = [f"procedural_{j}" for j in range(len(stage1_candidates))]

    # Determine candidates for this batch
    if archetypes:
        candidates      = [a["strategy"] for a in archetypes]
        candidate_names = [a["name"]     for a in archetypes]
    else:
        candidates      = stage1_candidates
        candidate_names = stage1_names

    # ── Generate all states and contexts upfront ──────────────────────────────
    scores                   = []
    archetype_wins           = {}
    archetype_wins_event     = {}
    archetype_wins_nonevent  = {}
    context_mix              = {}

    states   = [_SIM.random_state() for _ in range(n_rounds)]
    contexts = [build_context(s)    for s in states]

    # ── Score all rounds ──────────────────────────────────────────────────────
    score_args = [(state, candidates, candidate_names) for state in states]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(str(domain_path),),
        ) as pool:
            all_scored = list(pool.map(_score_round, score_args))
    else:
        all_scored = [_score_round(a) for a in score_args]

    # ── Process results sequentially ─────────────────────────────────────────
    from engine_extract import extract_principles_ai

    with open(log_path, "a") as log_file:
        for i, (scored, state, context) in enumerate(zip(all_scored, states, contexts)):
            round_num    = generation_offset + i + 1
            winner_score, winner, winner_name = scored[0]
            losers       = [c for _, c, _ in scored[1:4]]
            score_margin = round(scored[0][0] - scored[1][0], 4) if len(scored) > 1 else 0.0

            scores.append(winner_score)

            entry = {
                "round":        round_num,
                "score":        winner_score,
                "score_margin": score_margin,
                "metric":       _SIM.METRIC_NAME,
                "winner":       winner,
                "state":        state,
                "archetype":    winner_name,
            }
            log_file.write(json.dumps(entry) + "\n")

            # Track win counts
            if winner_name not in archetype_wins:
                archetype_wins[winner_name]          = 0
                archetype_wins_event[winner_name]    = 0
                archetype_wins_nonevent[winner_name] = 0
            archetype_wins[winner_name] += 1
            if is_event is not None and is_event(state):
                archetype_wins_event[winner_name] += 1
            else:
                archetype_wins_nonevent[winner_name] += 1

            # Track context distribution
            for k, v in context.items():
                key = f"{k}:{v}"
                context_mix[key] = context_mix.get(key, 0) + 1

            # Extract principles every 10 rounds
            if round_num % 10 == 0:
                playbook = extract_principles_ai(
                    winner=winner,
                    losers=losers,
                    context=context,
                    score=winner_score,
                    generation=round_num,
                    existing=playbook,
                    domain_path=domain_path,
                )
                _save_playbook(domain_path, playbook)

    # ── Champion propagation ──────────────────────────────────────────────────
    if archetypes and archetype_wins_nonevent:
        champion_name = max(archetype_wins_nonevent, key=archetype_wins_nonevent.get)
        champion = next((a for a in archetypes if a["name"] == champion_name), None)
        if champion:
            champion_data = {
                **champion,
                "nonevent_wins": archetype_wins_nonevent[champion_name],
                "total_wins":    archetype_wins[champion_name],
            }
            (domain_path / "champion_archetype.json").write_text(
                json.dumps(champion_data, indent=2)
            )

    # ── Save top candidates for Stage 1 evolution ────────────────────────────
    if archetypes:
        top_n = sorted(
            [
                (
                    archetype_wins_nonevent.get(a["name"], 0) * 2
                    + archetype_wins_event.get(a["name"], 0),
                    a["strategy"],
                    a["name"],
                )
                for a in archetypes
            ],
            key=lambda x: x[0],
            reverse=True,
        )[:4]
        top_candidates = [{"name": n, "strategy": c, "wins": w} for w, c, n in top_n]
        (domain_path / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))
    elif stage1_candidates:
        top_n = sorted(
            [
                (
                    archetype_wins_nonevent.get(stage1_names[j], 0)
                    + archetype_wins.get(stage1_names[j], 0),
                    stage1_candidates[j],
                    stage1_names[j],
                )
                for j in range(len(stage1_candidates))
            ],
            key=lambda x: x[0],
            reverse=True,
        )[:4]
        top_candidates = [{"name": n, "strategy": c, "wins": w} for w, c, n in top_n]
        (domain_path / "top_candidates.json").write_text(json.dumps(top_candidates, indent=2))

    # ── Build result dict for director ────────────────────────────────────────
    n       = len(scores)
    quarter = max(1, n // 4)
    first_q_avg = sum(scores[:quarter]) / quarter
    last_q_avg  = sum(scores[-quarter:]) / quarter
    trend_pct   = (
        (last_q_avg - first_q_avg) / abs(first_q_avg) * 100 if first_q_avg else 0.0
    )
    top_principles = sorted(
        playbook, key=lambda p: p.get("confidence", 0), reverse=True
    )[:5]

    return {
        "n_rounds":               n,
        "avg_score":              round(sum(scores) / n, 2) if n else 0,
        "best_score":             round(max(scores), 2)     if scores else 0,
        "worst_score":            round(min(scores), 2)     if scores else 0,
        "trend_pct":              round(trend_pct, 1),
        "score_last_10":          [round(s, 2) for s in scores[-10:]],
        "context_mix":            {
            k: v
            for k, v in sorted(context_mix.items(), key=lambda x: -x[1])[:10]
        },
        "playbook_size":          len(playbook),
        "top_principles":         top_principles,
        "archetype_wins":         archetype_wins,
        "archetype_wins_event":   archetype_wins_event,
        "archetype_wins_nonevent": archetype_wins_nonevent,
    }
