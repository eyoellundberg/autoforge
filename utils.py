"""
utils.py — File/data helpers. No AI calls.

Provides: ENGINE_ROOT, load_env, load_sim, load_world_model, load_hypotheses,
          normalize_confidence, normalize_playbook_entry, retire_principles,
          append_thinking_log, git_commit_batch.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ENGINE_ROOT = Path(__file__).parent


def load_env(domain_path: Path):
    """Load .env from domain folder, then engine root as fallback."""
    for env_file in [domain_path / ".env", ENGINE_ROOT / ".env"]:
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def load_sim(domain_path: Path):
    """Import the domain's simulation module. Raises ImportError on failure."""
    import importlib
    load_env(domain_path)
    if str(domain_path) not in sys.path:
        sys.path.insert(0, str(domain_path))
    os.chdir(domain_path)
    try:
        return importlib.import_module("simulation")
    except Exception as e:
        raise ImportError(f"simulation.py import failed: {e}") from e


def load_mission(domain_path: Path) -> str:
    """Return mission.md contents for a domain, if present."""
    mission_path = domain_path / "mission.md"
    if not mission_path.exists():
        return ""
    return mission_path.read_text().strip()


def load_world_model(domain_path: Path) -> str:
    """
    Return world_model.md contents. Falls back to assembling from legacy files
    (mission.md + prompts/brain.md + prompts/extract.md + prompts/director.md)
    for backward compatibility with existing domains.
    """
    wm_path = domain_path / "world_model.md"
    if wm_path.exists():
        return wm_path.read_text().strip()

    parts = []
    for label, path in [
        ("Domain Understanding", domain_path / "prompts" / "director.md"),
        ("Strategy Space", domain_path / "prompts" / "brain.md"),
        ("Extraction Guidance", domain_path / "prompts" / "extract.md"),
        ("Success Criteria", domain_path / "mission.md"),
    ]:
        if path.exists():
            parts.append(f"## {label}\n{path.read_text().strip()}")
    return "\n\n".join(parts) if parts else ""


def load_hypotheses(domain_path: Path) -> dict:
    """Load hypothesis state from hypotheses.json."""
    h_path = domain_path / "hypotheses.json"
    if h_path.exists():
        try:
            return json.loads(h_path.read_text())
        except Exception:
            pass
    return {"confirmed": [], "open": [], "tested": []}


def save_hypotheses(domain_path: Path, analysis: dict, prior_hypotheses: dict) -> dict:
    """
    Merge director hypothesis output into cumulative hypotheses.json.
    Confirmed hypotheses accumulate. Open hypotheses replace prior open list.
    """
    confirmed = list(set(
        prior_hypotheses.get("confirmed", []) +
        analysis.get("hypotheses_confirmed", [])
    ))
    open_hyp = analysis.get("hypotheses_open", [])
    tested = analysis.get("hypotheses_tested", [])

    payload = {"confirmed": confirmed, "open": open_hyp, "tested": tested}
    (domain_path / "hypotheses.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def normalize_confidence(value) -> float:
    """Normalize legacy confidence values like 73 into 0.73."""
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf > 1.0:
        conf /= 100.0
    return max(0.0, min(1.0, conf))


def normalize_playbook_entry(entry: dict) -> dict:
    """Return a playbook entry with normalized confidence."""
    return {**entry, "confidence": normalize_confidence(entry.get("confidence", 0))}


def retire_principles(analysis: dict, domain_path: Path):
    """Retire flagged playbook topics. Protects high-confidence principles (>=88%), caps at 2/batch."""
    to_retire = analysis.get("retire_principles", [])
    if not to_retire:
        return

    pb_path = domain_path / "playbook.jsonl"
    if not pb_path.exists():
        return

    entries = [normalize_playbook_entry(json.loads(l)) for l in pb_path.read_text().strip().splitlines() if l.strip()]
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


def append_thinking_log(log_path: Path, batch_num: int, result: dict, analysis: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    simulation_fix_section = ""
    if analysis.get("simulation_fix_suggestions"):
        fixes = chr(10).join(f"- {s}" for s in analysis["simulation_fix_suggestions"])
        simulation_fix_section = f"\n**Simulation fix suggestions:**\n{fixes}\n"

    hypotheses_section = ""
    if analysis.get("hypotheses_tested"):
        hypotheses_section += f"\n**Hypotheses tested:** {', '.join(analysis['hypotheses_tested'])}"
    if analysis.get("hypotheses_confirmed"):
        hypotheses_section += f"\n**Hypotheses confirmed:** {', '.join(analysis['hypotheses_confirmed'])}"
    if analysis.get("hypotheses_open"):
        hypotheses_section += f"\n**Open hypotheses:** {', '.join(analysis['hypotheses_open'])}"

    evolution_section = ""
    if analysis.get("simulation_patch_needed"):
        evolution_section += f"\n**Simulation patch needed:** {analysis.get('simulation_patch_rationale', '')}"
    schema_evo = analysis.get("schema_evolution", {})
    if schema_evo.get("add_parameters"):
        evolution_section += "\n**Schema additions proposed:** " + ", ".join(p["name"] for p in schema_evo["add_parameters"])
    if schema_evo.get("remove_parameters"):
        evolution_section += "\n**Schema removals proposed:** " + ", ".join(p["name"] for p in schema_evo["remove_parameters"])

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
{hypotheses_section}{simulation_fix_section}{evolution_section}

---"""

    with open(log_path, "a") as f:
        f.write(section + "\n")


def git_commit_batch(domain: str, domain_path: Path, global_batch: int, result: dict, analysis: dict):
    """
    Auto-commit domain state after each batch. Silently skips if git unavailable.
    Commit message: "{domain} batch {N}: avg {score} [{verdict}]"
    """
    import subprocess

    git_dir = ENGINE_ROOT / ".git"
    if not git_dir.exists():
        return

    stage_targets = [
        domain_path / "playbook.jsonl",
        domain_path / "retired_topics.json",
        domain_path / "champion_archetype.json",
        domain_path / "top_candidates.json",
        domain_path / "simulation.py",
        domain_path / "world_model.md",
        domain_path / "hypotheses.json",
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
        if result_proc.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", msg],
                cwd=ENGINE_ROOT, check=True, capture_output=True,
            )
    except Exception:
        pass
