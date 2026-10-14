import pandas as pd
from mlops_project.pipeline.training_pipeline import read_features


def test_single_feature_csv_preserves_header(tmp_path):
    path = tmp_path / "one.csv"
    pd.DataFrame({"alcohol": [10.5, 11.3]}).to_csv(path, index=False)
    X, y = read_features(path)
    assert list(X) == ["alcohol"] and len(X) == 2 and y is None
