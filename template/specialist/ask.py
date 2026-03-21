"""
Specialist — standalone inference. No Autoforge dependency.

Usage:
    from specialist.ask import ask, record

    response = ask({"basis": -0.35, "carry": 0.02, "season": "october"})
    # → "HOLD 70%. Wide carry + SA risk creates a window where holding
    #    4-6 weeks has ~75% probability of improving basis by $0.06-0.11."

    record(features, actual_outcome)   # log real outcome for retraining
"""

import json
from pathlib import Path

_DIR = Path(__file__).parent
_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx_lm not installed — run: pip install mlx mlx-lm")
        _model, _tokenizer = load(str(_DIR / "model"))
    return _model, _tokenizer


def _system_prompt() -> str:
    parts = ["You are a domain expert advisor. Give specific, actionable recommendations with numbers."]
    wm = _DIR / "world_model.md"
    if wm.exists():
        parts.append(wm.read_text().strip())
    pb = _DIR / "playbook.jsonl"
    if pb.exists():
        principles = [json.loads(l) for l in pb.read_text().splitlines() if l.strip()]
        top = sorted(principles, key=lambda p: p.get("confidence", 0), reverse=True)[:10]
        if top:
            lines = "\n".join(f"- {p['principle']}" for p in top)
            parts.append(f"Learned principles:\n{lines}")
    return "\n\n".join(parts)


def ask(features: dict | str, max_tokens: int = 512, thinking: bool = False) -> str:
    """
    Get a recommendation from the specialist.

    features: dict of domain inputs, or a plain string question
    returns:  natural language recommendation with reasoning and numbers
    """
    from mlx_lm import generate

    model, tokenizer = _load()

    if isinstance(features, dict):
        user_content = "\n".join(f"{k}: {v}" for k, v in features.items())
    else:
        user_content = str(features)

    # Qwen3 supports /think and /no_think tokens for hybrid reasoning mode
    if thinking:
        user_content = user_content + " /think"
    else:
        user_content = user_content + " /no_think"

    messages = [
        {"role": "system",  "content": _system_prompt()},
        {"role": "user",    "content": user_content},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


def record(features: dict, outcome: float):
    """Log a real outcome. Accumulated outcomes are used by retrain.py."""
    from datetime import datetime
    entry = {
        "features":  features,
        "outcome":   outcome,
        "timestamp": datetime.now().isoformat(),
    }
    with open(_DIR / "outcomes.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        features = json.loads(sys.argv[1])
    else:
        features = json.loads(input())
    print(ask(features))
