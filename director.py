"""
director.py — Director AI: schemas + call_director().

Analyzes each batch, issues a verdict, and steers the next batch.
"""

import json
import os
from pathlib import Path

from api import structured_ai_call
from utils import load_world_model, normalize_confidence

MODEL_DIRECTOR = os.environ.get("AUTOFORGE_DIRECTOR_MODEL", "claude-opus-4-6")


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
            "description": "Concrete fixes for simulation.py. Only populate for reward_hacking or needs_calibration.",
        },
        "breakthroughs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "principle": {"type": "string"},
                    "evidence":  {"type": "string"},
                    "reshapes":  {"type": "string"},
                },
                "required": ["principle", "evidence", "reshapes"],
                "additionalProperties": False,
            },
            "description": "Step-change discoveries that reshape how other principles should be interpreted. Not incremental improvements — only findings that change the model of the domain.",
        },
    },
    "required": [
        "verdict", "observations", "principles_gaining_confidence",
        "concerns", "mistakes_to_note", "next_batch_focus", "hints",
        "retire_principles", "simulation_fix_suggestions", "breakthroughs",
    ],
    "additionalProperties": False,
}


BOOTSTRAP_SCHEMA = {
    "type": "object",
    "properties": {
        "domain_summary":    {"type": "string"},
        "metric_name":       {"type": "string"},
        "simulation_py":     {"type": "string"},
        "world_model_md":    {"type": "string"},
        "context_keys":      {"type": "array", "items": {"type": "string"}},
        "calibration_notes": {"type": "string"},
    },
    "required": ["domain_summary", "metric_name", "simulation_py", "world_model_md",
                 "context_keys", "calibration_notes"],
    "additionalProperties": False,
}


def call_director(
    batch_num: int,
    result: dict,
    prior_analysis: dict,
    domain_path: Path,
    playbook_sizes: list,
) -> dict:
    """Analyze the batch and direct the next one."""

    world_model = load_world_model(domain_path)

    system_prompt = (
        "You are an expert analyst directing an autonomous learning engine. "
        "Be specific and actionable. Identify real learning vs simulation artifacts. "
        "Keep observations concise — 2-4 items each."
    )
    if world_model:
        system_prompt = "WORLD MODEL:\n" + world_model + "\n\n" + system_prompt

    top_principles = "\n".join(
        f"  [{normalize_confidence(p.get('confidence', 0)):.0%}] [{p.get('context', '')}] {p.get('principle', '')}"
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
        top_archetypes_str += "\nNOTE: wins marked 'event' may be sim artifacts. Weight non-event wins more heavily."

    prior = prior_analysis or {}
    prior_focus    = prior.get("next_batch_focus", "none — this is the first batch")
    prior_hints    = prior.get("hints", [])
    prior_mistakes = prior.get("mistakes_to_note", [])

    pb_path = domain_path / "playbook.jsonl"
    playbook_topics = "none"
    if pb_path.exists():
        topics = sorted({json.loads(l).get("topic", "") for l in pb_path.read_text().splitlines() if l.strip()})
        playbook_topics = ", ".join(topics) if topics else "none"

    growth_deltas = []
    if len(playbook_sizes) >= 2:
        growth_deltas = [playbook_sizes[i] - playbook_sizes[i - 1] for i in range(1, len(playbook_sizes))]
    growth_str = f"Playbook growth per batch: {growth_deltas}\n" if growth_deltas else ""
    saturation_note = ""
    if len(growth_deltas) >= 2 and growth_deltas[-1] == 0 and growth_deltas[-2] == 0:
        saturation_note = 'NOTE: if growth is 0 for 2+ batches and score is stable, use verdict "saturated"\n'

    prompt = f"""BATCH {batch_num} RESULTS:
  Rounds:     {result['n_rounds']}
  Avg score:  {result['avg_score']}
  Best score: {result['best_score']}
  Worst:      {result['worst_score']}
  Trend (first→last quarter): {result['trend_pct']:+.1f}%
  Last 10:    {result['score_last_10']}
  Context mix: {json.dumps(result['context_mix'])}
  Playbook:   {result['playbook_size']} principles
{growth_str}{saturation_note}
TOP PLAYBOOK PRINCIPLES:
{top_principles}
{top_archetypes_str}
PRIOR BATCH FOCUS: {prior_focus}
PRIOR HINTS: {json.dumps(prior_hints)}
PRIOR MISTAKES: {json.dumps(prior_mistakes)}

Analyze this batch. Is the engine learning real principles or gaming the simulation?
Is the score trending up meaningfully? Are principles plausible for this domain?
What should the next batch focus on?

Verdicts:
  converging        = score and playbook improving steadily
  exploring         = mixed results, still searching
  stalled           = no improvement for multiple batches
  reward_hacking    = score rising but for the wrong reason (sim artifact)
  needs_calibration = simulation needs adjustment before more runs
  saturated         = playbook full, score stable, nothing left to learn

Hints are short strings that bias strategy generation toward specific approaches.
Be specific — name what to explore and why.

retire_principles: playbook topic names to DELETE before the next batch.
Use when a principle is a confirmed sim artifact or contradicted by evidence.
Available topics: {playbook_topics}
GUARDRAIL: principles with confidence >=88% are protected. Cap retirements at 2 per batch.

simulation_fix_suggestions: REQUIRED for reward_hacking or needs_calibration.
Name the exact broken behavior and what to change. Empty [] otherwise.

breakthroughs: Step-change discoveries only — findings that reshape how other principles
should be interpreted, not just incremental score improvements. A breakthrough changes
the model of the domain. Most batches will have zero. Only flag genuine paradigm shifts.
For each: principle (what was discovered), evidence (score jump or pattern that confirms it),
reshapes (which existing principles this finding changes or overrides).
"""

    return structured_ai_call(
        task_name="director",
        domain_path=domain_path,
        model=MODEL_DIRECTOR,
        max_tokens=4096,
        system_prompt=system_prompt,
        user_prompt=prompt,
        schema=DIRECTOR_SCHEMA,
        metadata={"batch_num": batch_num, "playbook_size": result["playbook_size"]},
        thinking=True,
    )
