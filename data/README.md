# Wine Quality dataset

Primary source: [UCI Wine Quality, dataset 186](https://archive.ics.uci.edu/dataset/186/wine+quality).
DOI: [10.24432/C56S3T](https://doi.org/10.24432/C56S3T).

Creators: Paulo Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009).
Dataset license: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
Code licensing is separate from this dataset license.

## Download

```bash
python scripts/download_data.py
```

The script downloads the official archive and validates the expected SHA-256 fingerprint of each CSV.
Raw CSVs are kept local and ignored by Git. It fails if a source file changes rather than silently
training on a different dataset.

| CSV | SHA-256 |
|---|---|
| `winequality-red.csv` | `4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e` |
| `winequality-white.csv` | `76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836` |

## Contract

Each row is a wine sample with 11 numeric physicochemical measurements and the sensory target `quality`.
The target's stated range is 0–10. Red and white wine are evaluated separately in this repository.
Source columns: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide,
total sulfur dioxide, density, pH, sulphates, alcohol and quality. Files use a semicolon separator.

## Collection and evaluation limits

The historical samples describe Portuguese Vinho Verde wine. The source does not supply grape type,
brand or sale price. They do not establish current commercial drink quality. Exact full-row duplicates
are removed for this benchmark; identical measured rows may still represent separate samples. There
are no bottle/producer grouping IDs in the benchmark files, so independence between samples is unverified.
