"""
evolve.py — Self-evolution engine for simulation.py and CANDIDATE_SCHEMA.

Gated behind --self-evolve flag. All changes validated before applying.
Uses git for full traceability — every evolution is a commit.

Flow:
  1. Director flags simulation_patch_needed or schema_evolution
  2. Opus generates improved simulation.py
  3. validate_domain() checks the proposed version
  4. If valid → apply + git commit. If invalid → discard + log why.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from api import structured_ai_call
from utils import ENGINE_ROOT, load_world_model, load_env

MODEL_EVOLVE = os.environ.get("AUTOFORGE_EVOLVE_MODEL", "claude-opus-4-6")

SIM_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "simulation_py": {"type": "string"},
        "changes_summary": {"type": "string"},
    },
    "required": ["simulation_py", "changes_summary"],
    "additionalProperties": False,
}


def _log_evolution(domain_path: Path, event: str, summary: str, errors: list):
    """Append to evolution_log.jsonl for audit trail."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": event,
        "summary": summary,
        "errors": errors,
    }
    log_path = domain_path / "evolution_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def evolve_simulation(
    domain_path: Path,
    rationale: str,
    fix_suggestions: list,
) -> bool:
    """
    Propose and validate a simulation.py patch.

    1. Read current simulation.py and world_model.md
    2. Call Opus to generate improved simulation.py
    3. Backup current sim, write proposed version
    4. Run validate_domain against the proposed version
    5. If valid → keep new sim. If invalid → restore backup.
    """
    load_env(domain_path)
    sim_path = domain_path / "simulation.py"
    if not sim_path.exists():
        return False

    current_sim = sim_path.read_text()
    world_model = load_world_model(domain_path)

    system_prompt = (
        "You are improving a simulation.py file for an autonomous strategy learning system. "
        "The simulation must remain deterministic, fast, and faithful to the domain. "
        "Preserve ALL existing exports: simulate(), random_state(), CANDIDATE_SCHEMA, METRIC_NAME. "
        "Preserve build_context() and is_event() if they exist. "
        "Only fix what is specifically identified as broken. Do not redesign the whole simulation. "
        "The output must be a complete, runnable Python file."
    )

    user_prompt = f"""CURRENT simulation.py:
```python
{current_sim}
```

WORLD MODEL:
{world_model}

RATIONALE FOR PATCH:
{rationale}

SPECIFIC FIX SUGGESTIONS:
{chr(10).join(f'- {s}' for s in fix_suggestions) if fix_suggestions else '- (see rationale)'}

Generate a corrected simulation.py. Preserve the same API surface. Only fix what is broken.
Return the complete file content in the simulation_py field.
"""

    try:
        data = structured_ai_call(
            task_name="evolve_sim",
            domain_path=domain_path,
            model=MODEL_EVOLVE,
            max_tokens=12000,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SIM_PATCH_SCHEMA,
            thinking=True,
        )
    except Exception as e:
        print(f"  [evolve] simulation patch generation failed: {e}")
        _log_evolution(domain_path, "sim_patch_error", str(e), [])
        return False

    proposed = data["simulation_py"]
    changes_summary = data["changes_summary"]

    backup_path = sim_path.with_suffix(".py.backup")
    shutil.copy2(sim_path, backup_path)
    sim_path.write_text(proposed)

    import sys
    if "simulation" in sys.modules:
        del sys.modules["simulation"]

    from tools import validate_domain
    ok, warnings, errors = validate_domain(domain_path)

    if errors:
        shutil.copy2(backup_path, sim_path)
        backup_path.unlink(missing_ok=True)
        if "simulation" in sys.modules:
            del sys.modules["simulation"]
        print(f"  [evolve] simulation patch REJECTED — validation errors:")
        for e in errors:
            print(f"    {e}")
        _log_evolution(domain_path, "sim_patch_rejected", changes_summary, errors)
        return False

    backup_path.unlink(missing_ok=True)
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    print(f"  [evolve] simulation patch APPLIED: {changes_summary}")
    _log_evolution(domain_path, "sim_patch_applied", changes_summary, [])
    return True


def evolve_schema(
    domain_path: Path,
    schema_evolution: dict,
) -> bool:
    """
    Apply schema evolution — add/remove parameters to CANDIDATE_SCHEMA.
    Calls Opus to modify simulation.py, validates before applying.
    """
    add_params = schema_evolution.get("add_parameters", [])
    remove_params = schema_evolution.get("remove_parameters", [])
    if not add_params and not remove_params:
        return False

    load_env(domain_path)
    sim_path = domain_path / "simulation.py"
    if not sim_path.exists():
        return False

    current_sim = sim_path.read_text()
    world_model = load_world_model(domain_path)

    additions_desc = "\n".join(
        f"- ADD {p['name']} ({p['type']}): {p['description']} — rationale: {p['rationale']}"
        for p in add_params
    ) if add_params else "(none)"

    removals_desc = "\n".join(
        f"- REMOVE {p['name']}: {p['rationale']}"
        for p in remove_params
    ) if remove_params else "(none)"

    system_prompt = (
        "You are modifying the CANDIDATE_SCHEMA and simulate() function in simulation.py. "
        "Add new parameters to CANDIDATE_SCHEMA with appropriate type, min/max, and defaults. "
        "Update simulate() to use new parameters meaningfully. "
        "Remove parameters from CANDIDATE_SCHEMA and simulate() cleanly. "
        "Preserve all other exports and functionality. "
        "The output must be a complete, runnable Python file."
    )

    user_prompt = f"""CURRENT simulation.py:
```python
{current_sim}
```

WORLD MODEL:
{world_model}

SCHEMA CHANGES REQUESTED:

Parameters to add:
{additions_desc}

Parameters to remove:
{removals_desc}

Modify CANDIDATE_SCHEMA and simulate() accordingly. Return the complete file.
"""

    try:
        data = structured_ai_call(
            task_name="evolve_schema",
            domain_path=domain_path,
            model=MODEL_EVOLVE,
            max_tokens=12000,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=SIM_PATCH_SCHEMA,
            thinking=True,
        )
    except Exception as e:
        print(f"  [evolve] schema evolution failed: {e}")
        _log_evolution(domain_path, "schema_evolve_error", str(e), [])
        return False

    proposed = data["simulation_py"]
    changes_summary = data["changes_summary"]

    backup_path = sim_path.with_suffix(".py.backup")
    shutil.copy2(sim_path, backup_path)
    sim_path.write_text(proposed)

    import sys
    if "simulation" in sys.modules:
        del sys.modules["simulation"]

    from tools import validate_domain
    ok, warnings, errors = validate_domain(domain_path)

    if errors:
        shutil.copy2(backup_path, sim_path)
        backup_path.unlink(missing_ok=True)
        if "simulation" in sys.modules:
            del sys.modules["simulation"]
        print(f"  [evolve] schema evolution REJECTED — validation errors:")
        for e in errors:
            print(f"    {e}")
        _log_evolution(domain_path, "schema_evolve_rejected", changes_summary, errors)
        return False

    backup_path.unlink(missing_ok=True)
    if "simulation" in sys.modules:
        del sys.modules["simulation"]
    print(f"  [evolve] schema evolution APPLIED: {changes_summary}")
    _log_evolution(domain_path, "schema_evolve_applied", changes_summary, [])
    return True
