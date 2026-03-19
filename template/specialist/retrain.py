"""Retrain the specialist from real outcomes. No Autoforge needed."""

import csv
import json
import numpy as np
import xgboost as xgb
from pathlib import Path

_dir = Path(__file__).parent


def retrain():
    cfg = json.loads((_dir / "config.json").read_text())
    features = cfg["features"]
    params = cfg["params"]
    all_cols = features + params

    # Load original training data
    rows_X, rows_y = [], []
    csv_path = _dir.parent / "training_features.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = [float(row.get(c, 0)) for c in all_cols]
                    y = float(row.get("score", 0))
                    rows_X.append(x)
                    rows_y.append(y)
                except (ValueError, TypeError):
                    continue
        print(f"Loaded {len(rows_X)} rows from training_features.csv")

    # Load real outcomes
    outcomes_path = _dir / "outcomes.jsonl"
    n_real = 0
    if outcomes_path.exists():
        for line in outcomes_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                state = entry.get("state", {})
                decision = entry.get("decision", {})
                outcome = float(entry.get("outcome", 0))
                x = [float(state.get(f, 0)) for f in features] + \
                    [float(decision.get(p, 0)) for p in params]
                rows_X.append(x)
                rows_y.append(outcome)
                n_real += 1
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        print(f"Loaded {n_real} real outcomes from outcomes.jsonl")

    if not rows_X:
        print("No training data found.")
        return

    X = np.array(rows_X)
    y = np.array(rows_y)

    dtrain = xgb.DMatrix(X, label=y, feature_names=all_cols)
    xgb_params = {"max_depth": 4, "eta": 0.1, "objective": "reg:squarederror", "verbosity": 0}
    model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                      evals=[(dtrain, "train")], verbose_eval=50)

    model.save_model(str(_dir / "model.json"))

    preds = model.predict(dtrain)
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Update config
    cfg["trained_on"] = len(rows_X)
    cfg["real_outcomes"] = n_real
    cfg["r2"] = round(r2, 4)
    (_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"\nRetrained: {len(rows_X)} total examples ({n_real} real)")
    print(f"R²: {r2:.3f}")
    print(f"Saved: model.json")


if __name__ == "__main__":
    retrain()
