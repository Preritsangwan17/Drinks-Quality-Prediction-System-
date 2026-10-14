"""Train a wine-quality regression baseline on a local physicochemical CSV."""

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def read_features(path, training=False):
    frame = (
        pd.read_csv(path, sep=None, engine="python").drop_duplicates()
        if training
        else pd.read_csv(path, sep=None, engine="python")
    )
    if frame.empty:
        raise ValueError("CSV has no rows")
    if training and "quality" not in frame:
        raise ValueError("Training CSV requires the quality target")
    y = frame.pop("quality") if "quality" in frame else None
    X = frame.drop(columns=[c for c in frame if c.startswith("Unnamed:")])
    if X.shape[1] == 0:
        raise ValueError("CSV has no numeric features")
    try:
        X = X.apply(pd.to_numeric, errors="raise")
        if y is not None:
            y = pd.to_numeric(y, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("Features and quality must be numeric") from error
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Features cannot contain infinity")
    if training and (len(X) < 20 or y.isna().any() or not np.isfinite(y).all()):
        raise ValueError("Need at least 20 rows with finite quality targets")
    return X, y


def train(data, output):
    X, y = read_features(data, training=True)
    dev_X, test_X, dev_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    train_X, val_X, train_y, val_y = train_test_split(
        dev_X, dev_y, test_size=0.25, random_state=42
    )
    candidates = {
        "median_baseline": DummyRegressor(strategy="median"),
        "random_forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    models, errors = {}, {}
    for name, estimator in candidates.items():
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("regressor", estimator),
            ]
        )
        model.fit(train_X, train_y)
        models[name] = model
        errors[name] = float(mean_absolute_error(val_y, model.predict(val_X)))
    best = min(errors, key=errors.get)
    prediction = models[best].predict(test_X)
    metrics = {
        "selected_model": best,
        "validation_mae": errors,
        "test_mae": float(mean_absolute_error(test_y, prediction)),
        "test_rmse": float(root_mean_squared_error(test_y, prediction)),
        "test_r2": float(r2_score(test_y, prediction)),
        "seed": 42,
        "split_rows": {
            "train": len(train_y),
            "validation": len(val_y),
            "test": len(test_y),
        },
        "scikit_learn_version": sklearn.__version__,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": models[best], "features": list(X.columns), "metrics": metrics}, output
    )
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def predict(data, model_path, output):
    saved = joblib.load(model_path)  # Only load trusted artifacts.
    X, _ = read_features(data)
    missing = set(saved["features"]) - set(X)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    result = pd.read_csv(data, sep=None, engine="python")
    result["predicted_quality"] = saved["model"].predict(X[saved["features"]])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", default="artifacts/model.joblib")
    score = commands.add_parser("predict")
    score.add_argument("--input", required=True)
    score.add_argument("--model", default="artifacts/model.joblib")
    score.add_argument("--output", default="artifacts/predictions.csv")
    args = parser.parse_args()
    if args.command == "train":
        print(json.dumps(train(args.data, args.output), indent=2))
    else:
        print(f"Scored {len(predict(args.input, args.model, args.output))} rows")


if __name__ == "__main__":
    main()
