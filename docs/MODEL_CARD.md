# Model card: Wine sensory-quality regression

## Purpose and data

Predict the sensory `quality` score of Portuguese Vinho Verde wines from 11 physicochemical measurements.
Source: [UCI dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality), Cortez et al. (2009),
[DOI 10.24432/C56S3T](https://doi.org/10.24432/C56S3T), CC BY 4.0.
Red and white samples are separate experiments. This model does not evaluate arbitrary drinks or certify safety.

## Method

Remove exact measured-feature/target duplicates, split 60/20/20 with seed 42, fit median imputation on
training only. Compare a training-median predictor with Random Forest using validation MAE.
Evaluate the selected model and fixed baseline on the test partition after selection.
Outputs are continuous quality-score points, not rounded classes or calibrated probabilities.

## Evaluation and provenance

The README summarizes actual UCI red/white experiments; versioned reports contain fingerprints,
feature names, split sizes, validation MAE, held-out MAE/RMSE/R² and the test median baseline.
The official download script fails if an expected CSV fingerprint changes. Exact package versions
for the recorded experiment are included in each report. Model files are generated locally and ignored.

## Limits

Only physicochemical inputs and sensory outputs are available. No bottle/producer groups or collection
timestamps are used, so independent-group and future-time validation are unavailable. Removing identical
measurements assumes they are duplicates, which may not hold for every physical sample. Quality is ordinal;
regression treats score differences as numeric distances. No uncertainty intervals or external-site validation
have been completed. Do not apply a red-wine model to white wine without a separate transfer evaluation.

## Reproduction

```bash
python scripts/download_data.py
python app.py train --data data/winequality-red.csv --output artifacts/red_wine.joblib
python app.py predict --input examples/wine_transactions.csv --model artifacts/red_wine.joblib
```

Only load trusted Joblib artifacts. Illustrative example rows are not evaluation evidence.
