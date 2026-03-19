"""
director.py — Director AI: schemas + call_director().

The director analyzes each batch, issues a verdict, tracks hypotheses,
and steers the next batch. Uses Opus with adaptive thinking.
"""

import json
import os
from pathlib import Path

from api import structured_ai_call
from utils import load_world_model, normalize_confidence, load_hypotheses

MODEL_DIRECTOR = os.environ.get("AUTOFORGE_DIRECTOR_MODEL", "claude-opus-4-6")


# ── Director schema ──────────────────────────────────────────────────────────

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
            "description": "Concrete suggestions for fixing simulation.py. Empty unless reward_hacking or needs_calibration.",
        },
        "hypotheses_tested": {
            "type": "array", "items": {"type": "string"},
        },
        "hypotheses_confirmed": {
            "type": "array", "items": {"type": "string"},
        },
        "hypotheses_open": {
            "type": "array", "items": {"type": "string"},
        },
        "simulation_patch_needed": {
            "type": "boolean",
            "description": "True only for reward_hacking or needs_calibration with a specific diagnosis.",
        },
        "simulation_patch_rationale": {
            "type": "string",
            "description": "Why the simulation needs patching. Empty if simulation_patch_needed is false.",
        },
        "schema_evolution": {
            "type": "object",
            "properties": {
                "add_parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":        {"type": "string"},
                            "type":        {"type": "string"},
                            "description": {"type": "string"},
                            "rationale":   {"type": "string"},
                        },
                        "required": ["name", "type", "description", "rationale"],
                        "additionalProperties": False,
                    },
                },
                "remove_parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":      {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["name", "rationale"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["add_parameters", "remove_parameters"],
            "additionalProperties": False,
            "description": "Propose schema changes only with strong evidence. Empty arrays when no changes needed.",
        },
    },
    "required": [
        "verdict", "observations", "principles_gaining_confidence",
        "concerns", "mistakes_to_note", "next_batch_focus", "hints",
        "retire_principles", "simulation_fix_suggestions",
        "hypotheses_tested", "hypotheses_confirmed", "hypotheses_open",
        "simulation_patch_needed", "simulation_patch_rationale",
        "schema_evolution",
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
        "world_model_md":    {"type": "string"},
        "context_keys":      {"type": "array", "items": {"type": "string"}},
        "calibration_notes": {"type": "string"},
    },
    "required": ["domain_summary", "metric_name", "simulation_py", "world_model_md",
                 "context_keys", "calibration_notes"],
    "additionalProperties": False,
}


# ── Director call ─────────────────────────────────────────────────────────────

def call_director(
    batch_num: int,
    result: dict,
    prior_analysis: dict,
    all_results: list,
    domain_path: Path,
    playbook_sizes: list,
) -> dict:
    """Ask the director AI to analyze the batch and direct the next one."""

    world_model = load_world_model(domain_path)

    system_prompt = (
        "You are an expert analyst directing an autonomous learning engine. "
        "Be specific and actionable. Identify real learning vs sim artifacts. "
        "Keep observations concise — 2-4 items each."
    )
    if world_model:
        system_prompt = "WORLD MODEL:\n" + world_model + "\n\n" + system_prompt

    batch_avgs = [r["avg_score"] for r in all_results]
    overall_trend = ""
    if len(batch_avgs) >= 2:
        delta = batch_avgs[-1] - batch_avgs[0]
        overall_trend = f"{delta:+.2f} from batch 1 to batch {batch_num}"

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
        top_archetypes_str += (
            "\nNOTE: wins marked 'event' may be sim artifacts. Weight non-event wins more heavily."
        )

    prior = prior_analysis or {}
    prior_focus    = prior.get("next_batch_focus", "none — this is the first batch")
    prior_hints    = prior.get("hints", [])
    prior_mistakes = prior.get("mistakes_to_note", [])

    hypotheses = load_hypotheses(domain_path)
    hypotheses_section = ""
    if hypotheses.get("open"):
        hypotheses_section += "\nOPEN HYPOTHESES FROM PRIOR BATCH:\n" + "\n".join(f"  - {h}" for h in hypotheses["open"])
    if hypotheses.get("confirmed"):
        hypotheses_section += "\nCONFIRMED HYPOTHESES:\n" + "\n".join(f"  - {h}" for h in hypotheses["confirmed"][-5:])

    pb_path = domain_path / "playbook.jsonl"
    playbook_topics = "none"
    if pb_path.exists():
        topics = sorted({json.loads(l).get("topic", "") for l in pb_path.read_text().splitlines() if l.strip()})
        playbook_topics = ", ".join(topics) if topics else "none"

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
{top_archetypes_str}{hypotheses_section}
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
Be specific: name the exact behavior that is broken and what to change in simulate() or random_state().
Leave empty [] for converging / exploring / stalled / saturated.

HYPOTHESIS TRACKING:
- hypotheses_tested: what hypotheses were being tested this batch
- hypotheses_confirmed: what was confirmed by the results (clear evidence)
- hypotheses_open: what remains unclear and should be tested next batch

SIMULATION EVOLUTION (simulation_patch_needed + simulation_patch_rationale):
Set simulation_patch_needed to true ONLY when reward_hacking or needs_calibration is detected
and you have a specific diagnosis of what is wrong with simulation.py.

SCHEMA EVOLUTION (schema_evolution):
Propose add_parameters when tournament results suggest a missing strategic dimension.
Propose remove_parameters when a parameter has zero effect on scores across multiple batches.
Leave both arrays empty unless you have strong evidence from the tournament results.
"""

    return structured_ai_call(
        task_name="director",
        domain_path=domain_path,
        model=MODEL_DIRECTOR,
        max_tokens=8000,
        system_prompt=system_prompt,
        user_prompt=prompt,
        schema=DIRECTOR_SCHEMA,
        metadata={"batch_num": batch_num, "playbook_size": result["playbook_size"]},
        thinking=True,
    )
