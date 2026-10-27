# MLflow Experiment Tracking and Data Drift Analysis

Tracks a Random Forest experiment with MLflow and detects simulated data drift on the
Breast Cancer Wisconsin (Diagnostic) dataset, using the KS test and PSI.

## Tested environment
- Windows 11, Python 3.11.9, MLflow 3.17.2

## Dataset
Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn (569 records, 30 numeric
features, binary target: malignant / benign). No download is needed.

## How to run
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python drift_analysis.py
mlflow ui                      # open http://127.0.0.1:5000
```

## What the script does
1. Splits data into reference (60%) and current (40%) sets.
2. Simulates drift on the current set: `mean radius` x1.20, `mean texture` +4 (plus noise), `mean area` x1.45.
3. Trains a Random Forest on the reference data.
4. Logs parameters, metrics, the model and artifacts to the MLflow experiment `BreastCancer_Drift_Monitoring`.
5. Runs a KS test and PSI on all 30 features. A feature counts as drifted when p < 0.05 and PSI > 0.2.

## Outputs (in `outputs/` and in MLflow artifacts)
- `drift_report.csv`: KS statistic, p-value, PSI and drift flag per feature
- `feature_distributions.png`: reference vs current histograms of the drifted features
- `psi_top10.png`: top 10 features by PSI

## Results (seed = 42)
- Drift detected in 3 of 30 features: mean texture, mean area, mean radius
- Accuracy: 0.977 on reference test data, 0.930 on drifted current data
