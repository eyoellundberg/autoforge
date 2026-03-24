"""
Retrain the specialist on real outcomes. No Playbook-ml needed.

Run manually:   python specialist/retrain.py
Or via cron:    0 2 * * * cd /your/app && python specialist/retrain.py
"""

import json
import subprocess
import sys
from pathlib import Path

_DIR = Path(__file__).parent
OUTCOMES_PATH = _DIR / "outcomes.jsonl"
MIN_OUTCOMES = 50


def _outcomes() -> list:
    if not OUTCOMES_PATH.exists():
        return []
    items = []
    for line in OUTCOMES_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def _make_training_examples(outcomes: list) -> Path:
    """Convert real outcomes into fine-tuning JSONL."""
    train_path = _DIR / "retrain_data.jsonl"
    with open(train_path, "w") as f:
        for o in outcomes:
            features = o.get("features", {})
            outcome  = o.get("outcome", 0)
            user_content = "\n".join(f"{k}: {v}" for k, v in features.items())
            assistant_content = f"Outcome: {outcome}"
            example = {
                "messages": [
                    {"role": "user",      "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
            }
            f.write(json.dumps(example) + "\n")
    return train_path


def retrain():
    outcomes = _outcomes()
    n = len(outcomes)

    if n < MIN_OUTCOMES:
        print(f"{n} outcomes logged — need {MIN_OUTCOMES}+ to retrain. Recording more outcomes.")
        return

    print(f"Retraining on {n} real outcomes...")

    train_path = _make_training_examples(outcomes)
    adapter_path = _DIR / "retrain_adapters"
    model_path   = _DIR / "model"

    if not model_path.exists():
        print(f"Model not found at {model_path} — run playbook-ml train first.")
        sys.exit(1)

    # Fine-tune with LoRA (200 iterations — fast update, not a full retrain)
    try:
        subprocess.run([
            sys.executable, "-m", "mlx_lm.lora",
            "--model", str(model_path),
            "--train",
            "--data", str(_DIR),
            "--iters", "200",
            "--batch-size", "2",
            "--adapter-path", str(adapter_path),
            "--train-file", train_path.name,
        ], check=True, cwd=str(_DIR))
    except FileNotFoundError:
        print("mlx_lm not found — install with: pip install mlx mlx-lm")
        sys.exit(1)

    # Fuse adapter back into model (updates model in place)
    subprocess.run([
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(model_path),
        "--adapter-path", str(adapter_path),
        "--save-path", str(model_path),
    ], check=True)

    print(f"Retrained on {n} real outcomes. Model updated.")


if __name__ == "__main__":
    retrain()
