"""
commands/shared.py — Shared constants, helpers, and AI-call utilities.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

ENGINE_ROOT = Path(__file__).parent.parent
console = Console()


def load_env(domain_path: Path):
    """Load .env from domain folder, then engine root as fallback."""
    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


MODEL_DIRECTOR = os.environ.get("AUTOFORGE_DIRECTOR_MODEL", "claude-sonnet-4-6")


# ── Director schema ─────────────────────────────────────────────────────────

DIRECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["converging", "exploring", "stalled", "reward_hacking", "needs_calibration", "saturated"],
        },
        "observations": {
            "type": "array", "items": {"type": "string"},
        },
        "principles_gaining_confidence": {
            "type": "array", "items": {"type": "string"},
        },
        "concerns": {
            "type": "array", "items": {"type": "string"},
        },
        "mistakes_to_note": {
            "type": "array", "items": {"type": "string"},
        },
        "next_batch_focus": {"type": "string"},
        "hints": {
            "type": "array", "items": {"type": "string"},
        },
        "retire_principles": {
            "type": "array", "items": {"type": "string"},
            "description": "Topic names to remove from the playbook before the next batch.",
        },
        "simulation_fix_suggestions": {
            "type": "array", "items": {"type": "string"},
            "description": "Concrete, actionable suggestions for fixing simulation.py when reward_hacking or needs_calibration is detected. Empty otherwise.",
        },
    },
    "required": [
        "verdict", "observations", "principles_gaining_confidence",
        "concerns", "mistakes_to_note", "next_batch_focus", "hints",
        "retire_principles", "simulation_fix_suggestions",
    ],
    "additionalProperties": False,
}


# ── Bootstrap schema ─────────────────────────────────────────────────────────

BOOTSTRAP_SCHEMA = {
    "type": "object",
    "properties": {
        "domain_summary":    {"type": "string"},
        "metric_name":       {"type": "string"},
        "simulation_py":     {"type": "string"},
        "brain_md":          {"type": "string"},
        "extract_md":        {"type": "string"},
        "director_md":       {"type": "string"},
        "context_keys":      {"type": "array", "items": {"type": "string"}},
        "calibration_notes": {"type": "string"},
    },
    "required": ["domain_summary", "metric_name", "simulation_py", "brain_md",
                 "extract_md", "director_md", "context_keys", "calibration_notes"],
    "additionalProperties": False,
}


# ── Director call ────────────────────────────────────────────────────────────

def call_director(
    batch_num: int,
    result: dict,
    prior_analysis: dict,
    all_results: list,
    domain_path: Path,
    playbook_sizes: list,
) -> dict:
    """Ask the director AI to analyze the batch and direct the next one."""

    director_md = domain_path / "prompts" / "director.md"
    domain_context = director_md.read_text().strip() if director_md.exists() else ""

    system_prompt = (
        "You are an expert analyst directing an autonomous learning engine. "
        "Be specific and actionable. Identify real learning vs sim artifacts. "
        "Keep observations concise — 2-4 items each."
    )
    if domain_context:
        system_prompt = domain_context + "\n\n" + system_prompt

    # Trend across all batches so far
    batch_avgs = [r["avg_score"] for r in all_results]
    overall_trend = ""
    if len(batch_avgs) >= 2:
        delta = batch_avgs[-1] - batch_avgs[0]
        overall_trend = f"{delta:+.2f} from batch 1 to batch {batch_num}"

    top_principles = "\n".join(
        f"  [{p.get('confidence', 0):.0%}] [{p.get('context', '')}] {p.get('principle', '')}"
        for p in result["top_principles"]
    ) or "  none yet"

    archetype_wins          = result.get("archetype_wins", {})
    archetype_wins_event    = result.get("archetype_wins_event", {})
    archetype_wins_nonevent = result.get("archetype_wins_nonevent", {})
    top_archetypes_str = ""
    if archetype_wins:
        top = sorted(archetype_wins.items(), key=lambda x: x[1], reverse=True)[:6]
        top_archetypes_str = "\nTOP WINNING ARCHETYPES THIS BATCH:\n" + "\n".join(
            f"  {name}: {count} total  ({archetype_wins_event.get(name, 0)} event / {archetype_wins_nonevent.get(name, 0)} non-event)"
            for name, count in top
        )
        top_archetypes_str += (
            "\nNOTE: wins marked 'event' may be sim artifacts (demand is guaranteed when event fires)."
            " Weight non-event wins more heavily when evaluating archetype quality."
        )

    prior_focus   = prior_analysis.get("next_batch_focus", "none — this is the first batch") if prior_analysis else "none — this is the first batch"
    prior_hints   = prior_analysis.get("hints", []) if prior_analysis else []
    prior_mistakes = prior_analysis.get("mistakes_to_note", []) if prior_analysis else []

    pb_path = domain_path / "playbook.jsonl"
    playbook_topics = "none"
    if pb_path.exists():
        topics = sorted({json.loads(l).get("topic", "") for l in pb_path.read_text().splitlines() if l.strip()})
        playbook_topics = ", ".join(topics) if topics else "none"

    # Playbook growth deltas
    growth_deltas = []
    if len(playbook_sizes) >= 2:
        growth_deltas = [playbook_sizes[i] - playbook_sizes[i - 1] for i in range(1, len(playbook_sizes))]
    growth_str = f"Playbook growth per batch: {growth_deltas}" if growth_deltas else ""
    saturation_note = ""
    if len(growth_deltas) >= 2 and growth_deltas[-1] == 0 and growth_deltas[-2] == 0:
        saturation_note = '\nNOTE: if growth is 0 for 2+ batches and score is stable, use verdict "saturated"'

    prompt = f"""You are directing an autonomous learning engine.

BATCH {batch_num} RESULTS:
  Rounds:        {result['n_rounds']}
  Avg score:     {result['avg_score']}
  Best score:    {result['best_score']}
  Worst score:   {result['worst_score']}
  Trend (first→last quarter): {result['trend_pct']:+.1f}%
  Overall across batches: {overall_trend or 'n/a'}
  Last 10 rounds scores: {result['score_last_10']}
  Context mix: {json.dumps(result['context_mix'])}
  Playbook size: {result['playbook_size']} principles
{growth_str}{saturation_note}

TOP PLAYBOOK PRINCIPLES:
{top_principles}
{top_archetypes_str}
PRIOR BATCH FOCUS: {prior_focus}
PRIOR HINTS APPLIED: {json.dumps(prior_hints)}
MISTAKES NOTED PREVIOUSLY: {json.dumps(prior_mistakes)}

Analyze this batch. Is the engine learning real principles or gaming the sim?
Is the score trending up meaningfully or flat? Are the playbook principles plausible?
What should the next batch focus on to push learning further?

Verdicts:
  converging         = score and playbook are improving steadily
  exploring          = mixed results, still searching
  stalled            = no improvement for multiple batches
  reward_hacking     = sim artifact — score rising but for the wrong reason
  needs_calibration  = sim or prompts need adjustment before more runs
  saturated          = playbook full, score stable, nothing left to learn from this sim

Hints are short strings that bias strategy generation toward certain approaches.
Good hints are specific and actionable — name the thing to explore and why.

retire_principles is a list of playbook topic names to DELETE before the next batch.
Use it when a principle is a confirmed sim artifact, suppressing real exploration, or contradicted by evidence.
Available topics in the current playbook: {playbook_topics}
GUARDRAIL: principles with confidence >=88% are protected and cannot be retired — do not list them.
Cap retirements at 2 per batch. Only retire if you have clear evidence the principle is wrong or harmful.

simulation_fix_suggestions: REQUIRED when verdict is reward_hacking or needs_calibration.
The user may not know what is wrong with their simulation.py — you are their expert diagnosis.
Be specific: name the exact behavior that is broken and what to change in simulate() or random_state().
Examples of good suggestions:
  - "simulate() always returns positive scores even when the strategy should fail — add a penalty branch for [CONDITION]"
  - "random_state() never generates [SCENARIO TYPE] — add a case so strategies are tested under that condition"
  - "score range is 0.95–1.05 — too narrow for strategies to differentiate; widen the reward spread"
  - "one parameter (e.g. threshold) has no effect on score — simulate() may not be reading it"
Leave empty [] for converging / exploring / stalled / saturated.
"""

    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL_DIRECTOR,
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": DIRECTOR_SCHEMA,
            }
        },
    )

    return json.loads(response.content[0].text)


# ── Thinking log ─────────────────────────────────────────────────────────────

def append_thinking_log(log_path: Path, batch_num: int, result: dict, analysis: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    section = f"""
## Batch {batch_num} — {timestamp}  |  {result['n_rounds']} rounds  |  avg score {result['avg_score']}  |  trend {result['trend_pct']:+.1f}%

**Verdict:** {analysis['verdict']}

**Observations:**
{chr(10).join(f"- {o}" for o in analysis['observations'])}

**Principles gaining confidence:**
{chr(10).join(f"- {p}" for p in analysis['principles_gaining_confidence']) or "- none yet"}

**Concerns:**
{chr(10).join(f"- {c}" for c in analysis['concerns']) or "- none"}

**Mistakes to not repeat:**
{chr(10).join(f"- {m}" for m in analysis['mistakes_to_note']) or "- none"}

**Next batch focus:** {analysis['next_batch_focus']}

**Hints injected for next batch:**
{chr(10).join(f"- {h}" for h in analysis['hints']) or "- none"}

{("**Simulation fix suggestions:**\n" + chr(10).join(f"- {s}" for s in analysis.get('simulation_fix_suggestions', []))) if analysis.get('simulation_fix_suggestions') else ""}

---"""

    with open(log_path, "a") as f:
        f.write(section + "\n")


# ── Git commit helper ─────────────────────────────────────────────────────────

def git_commit_batch(domain: str, domain_path: Path, global_batch: int, result: dict, analysis: dict):
    """
    Auto-commit domain state after each batch. Silently skips if git unavailable.

    Commits: playbook.jsonl, retired_topics.json, champion_archetype.json,
             top_candidates.json, simulation.py, tournament.py, prompts/

    Commit message: "{domain} batch {N}: avg {score} [{verdict}]"
    Each kept commit = one experiment. git log = full experiment history.
    """
    import subprocess

    git_dir = ENGINE_ROOT / ".git"
    if not git_dir.exists():
        return  # not a git repo — skip silently

    # Files to stage: learned state + domain code (not artifacts)
    stage_targets = [
        domain_path / "playbook.jsonl",
        domain_path / "retired_topics.json",
        domain_path / "champion_archetype.json",
        domain_path / "top_candidates.json",
        domain_path / "simulation.py",
        domain_path / "tournament.py",
        domain_path / "prompts",
    ]
    existing = [str(p.relative_to(ENGINE_ROOT)) for p in stage_targets if p.exists()]
    if not existing:
        return

    verdict   = analysis["verdict"]
    avg_score = result["avg_score"]
    focus     = analysis.get("next_batch_focus", "")[:60]

    msg_lines = [
        f"{domain} batch {global_batch}: avg {avg_score} [{verdict}]",
        focus,
    ]
    if analysis.get("concerns"):
        msg_lines.append(f"concern: {analysis['concerns'][0][:80]}")
    msg = "\n".join(l for l in msg_lines if l)

    try:
        subprocess.run(
            ["git", "add"] + existing,
            cwd=ENGINE_ROOT, check=True, capture_output=True,
        )
        result_proc = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=ENGINE_ROOT, capture_output=True,
        )
        if result_proc.returncode != 0:  # staged changes exist
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=ENGINE_ROOT, check=True, capture_output=True,
            )
    except Exception:
        pass  # git failure never interrupts a run


# ── Retire principles helper ──────────────────────────────────────────────────

def retire_principles(analysis: dict, domain_path: Path):
    """Retire flagged playbook topics. Protects high-confidence principles (>=88%), caps at 2/batch."""
    to_retire = analysis.get("retire_principles", [])
    if not to_retire:
        return

    pb_path = domain_path / "playbook.jsonl"
    if not pb_path.exists():
        return

    entries = [json.loads(l) for l in pb_path.read_text().strip().splitlines() if l.strip()]
    protected = {e.get("topic") for e in entries if e.get("confidence", 0) >= 0.88}
    safe_to_retire = [t for t in to_retire if t not in protected][:2]

    if safe_to_retire:
        kept = [e for e in entries if e.get("topic") not in safe_to_retire]
        retired_count = len(entries) - len(kept)
        with open(pb_path, "w") as f:
            for e in kept:
                f.write(json.dumps(e) + "\n")
        if retired_count:
            print(f"  retired {retired_count} principle(s): {', '.join(safe_to_retire)}")

        rt_path = domain_path / "retired_topics.json"
        rt_list = json.loads(rt_path.read_text()) if rt_path.exists() else []
        for t in safe_to_retire:
            if t not in rt_list:
                rt_list.append(t)
        rt_path.write_text(json.dumps(rt_list))

    blocked = [t for t in to_retire if t in protected]
    if blocked:
        print(f"  protected (>=88% conf): {', '.join(blocked)}")
