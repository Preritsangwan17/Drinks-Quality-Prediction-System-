import numpy as np
import pandas as pd
import pytest
from mlops_project.pipeline.training_pipeline import read_features, train, predict


def test_semicolon_training_and_prediction_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.uniform(8, 15, 120)
    data = tmp_path / "wine.csv"
    pd.DataFrame(
        {"alcohol": x, "acidity": rng.uniform(3, 9, 120), "quality": 0.5 * x}
    ).to_csv(data, sep=";", index=False)
    artifact = tmp_path / "model.joblib"
    metrics = train(data, artifact)
    assert metrics["selected_model"] == "random_forest"
    assert metrics["split_rows"] == {"train": 72, "validation": 24, "test": 24}
    pd.DataFrame({"alcohol": [10, 10], "acidity": [5, 5]}).to_csv(data, index=False)
    results = predict(data, artifact, tmp_path / "predictions.csv")
    assert len(results) == 2  # prediction must preserve duplicate transactions
    assert np.isfinite(results["predicted_quality"]).all()
    pd.DataFrame({"alcohol": [10]}).to_csv(data, index=False)
    with pytest.raises(ValueError, match="Missing feature"):
        predict(data, artifact, tmp_path / "invalid.csv")


def test_missing_target(tmp_path):
    data = tmp_path / "bad.csv"
    pd.DataFrame({"alcohol": [1, 2], "acidity": [3, 4]}).to_csv(data, index=False)
    with pytest.raises(ValueError, match="quality target"):
        read_features(data, training=True)
