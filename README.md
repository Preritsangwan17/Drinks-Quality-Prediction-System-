![Drinks Quality Prediction](docs/assets/banner.svg)

[![CI](https://github.com/Preritsangwan17/Drinks-Quality-Prediction-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/Preritsangwan17/Drinks-Quality-Prediction-System-/actions/workflows/ci.yml)

# Drinks Quality Prediction

A command-line machine-learning baseline for **regression**, with reproducible training, held-out evaluation,
model serialization and batch prediction. Previously an empty project scaffold; the repository now
contains an implemented baseline and software tests.

**Status:** baseline implementation. No real-data model performance is claimed. A verified training
dataset and dataset-specific evaluation are still required.

## What it does

The pipeline compares a median DummyRegressor with a Random Forest regressor, selects by lowest
validation MAE, then reports test MAE, RMSE and R². Quality is treated as numeric regression: predictions
are continuous scores. Comma and semicolon CSVs are accepted; all feature columns must be numeric.
Missing features are median-imputed using training data only.

Both experiments use seed 42 and 60% training / 20% validation / 20% test partitions after deduplication.
The selected model remains fitted on the training partition so the validation operating point remains
consistent with evaluation. Training writes an artifact and an adjacent `.metrics.json` report.

## Quick start

Use Python 3.11 or 3.12:

```bash
git clone https://github.com/Preritsangwan17/Drinks-Quality-Prediction-System-.git
cd Drinks-Quality-Prediction-System-
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py --help
```

Add the documented CSV locally, then run:

```bash
python app.py train --data data/winequality-red.csv
python app.py predict --input data/new_records.csv --output artifacts/predictions.csv
```

The prediction CSV must contain the training feature columns; its target is optional.
Generated models, metrics and predictions live in `artifacts/` and are ignored by Git.
Only load serialized models from trusted sources.

## Repository guide

| Path | Purpose |
|---|---|
| `mlops_project/pipeline/training_pipeline.py` | Validation, preprocessing, training, evaluation and prediction |
| `app.py` | CLI entry point |
| `data/README.md` | Dataset contract and provenance requirements |
| `tests/` | Synthetic artifact round-trip and invalid-input tests |
| `.github/workflows/ci.yml` | Automated software tests |

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use explicitly synthetic data to verify executable behavior, schema checks and serialization.
They do not establish real-world model accuracy. See [contribution guidance](CONTRIBUTING.md).

## Limitations and development priorities

Predicts the sensory-quality score of wine from numeric physicochemical measurements. It does not assess every drink, certify safety or replace laboratory testing.

Before claiming predictive performance, document the data source and license, inspect missing values,
duplicate entities and leakage, choose a temporal or grouped split if needed, and publish a real-data
report alongside a simple baseline. Random row splitting is only an initial experiment. Add calibrated
scores or interval estimates where the application requires them.

## License and author

Code is covered by the existing [MIT license](LICENSE). Dataset terms are separate.
Built by [Prerit Sangwan](https://github.com/Preritsangwan17), B.Tech CSE (AI/ML).
