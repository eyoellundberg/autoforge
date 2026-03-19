"""Specialist — standalone inference. No Autoforge dependency."""

import json
import numpy as np
import xgboost as xgb
from pathlib import Path

_dir = Path(__file__).parent
_cfg = json.loads((_dir / "config.json").read_text())

_model = xgb.Booster()
_model.load_model(str(_dir / "model.json"))

_FEATURES = _cfg["features"]
_PARAMS = _cfg["params"]
_RANGES = _cfg["param_ranges"]
_THRESHOLD = _cfg["abstain_threshold"]


def predict(scenario: dict, n_candidates: int = 50) -> dict:
    """
    Score candidate strategies against a scenario, return the best one.

    Returns {"strategy": {...}, "score": float}
    or      {"action": "ABSTAIN", "reason": "..."}
    """
    candidates = [
        {p: float(np.random.uniform(r[0], r[1])) for p, r in _RANGES.items()}
        for _ in range(n_candidates)
    ]
    rows = np.array([
        [float(scenario.get(f, 0)) for f in _FEATURES]
        + [c[p] for p in _PARAMS]
        for c in candidates
    ])
    scores = _model.predict(xgb.DMatrix(rows, feature_names=_FEATURES + _PARAMS))

    best_idx = int(np.argmax(scores))
    margin = float(scores[best_idx] - np.median(scores))

    if margin < _THRESHOLD:
        return {"action": "ABSTAIN", "reason": "strategies too close — escalate"}
    return {"strategy": candidates[best_idx], "score": float(scores[best_idx])}


def record(scenario: dict, decision: dict, outcome: float):
    """Record a real outcome for future retraining."""
    with open(_dir / "outcomes.jsonl", "a") as f:
        f.write(json.dumps({
            "state": scenario, "decision": decision, "outcome": outcome,
        }) + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scenario = json.loads(sys.argv[1])
    else:
        scenario = json.loads(input())
    print(json.dumps(predict(scenario), indent=2))
