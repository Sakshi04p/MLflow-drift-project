"""
MLflow Experiment Tracking + Data Drift Analysis
Dataset : Breast Cancer Wisconsin (Diagnostic) - bundled with scikit-learn
Drift   : Simulated data drift (covariate shift) on selected features
Method  : Kolmogorov-Smirnov test + Population Stability Index (PSI)
"""
import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from scipy.stats import ks_2samp
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

SEED = 42
KS_ALPHA = 0.05      # drift if p-value < 0.05
PSI_THRESHOLD = 0.2  # drift if PSI > 0.2 (common industry rule of thumb)
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


def psi(reference, current, bins=10):
    """Population Stability Index between two 1-D samples."""
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------- 1. data
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
X_ref, X_cur, y_ref, y_cur = train_test_split(
    X, y, test_size=0.4, random_state=SEED, stratify=y
)
X_train, X_test, y_train, y_test = train_test_split(
    X_ref, y_ref, test_size=0.25, random_state=SEED, stratify=y_ref
)

# Simulated drift: scanner/lab calibration change on 3 features of the current data
rng = np.random.default_rng(SEED)
X_cur_drift = X_cur.copy()
drifted_features = ["mean radius", "mean texture", "mean area"]
X_cur_drift["mean radius"] *= 1.20
X_cur_drift["mean texture"] += 4.0 + rng.normal(0, 0.5, len(X_cur_drift))
X_cur_drift["mean area"] *= 1.45

# ------------------------------------------------------------ 2. mlflow run
mlflow.set_experiment("BreastCancer_Drift_Monitoring")

with mlflow.start_run(run_name="rf_baseline_and_drift_check"):
    # ---- parameters
    params = {"model": "RandomForestClassifier", "n_estimators": 100,
              "max_depth": 6, "random_state": SEED,
              "ks_alpha": KS_ALPHA, "psi_threshold": PSI_THRESHOLD,
              "drifted_features": ",".join(drifted_features)}
    mlflow.log_params(params)

    # ---- train model
    model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
    model.fit(X_train, y_train)

    # ---- performance: reference test vs drifted current data
    pred_test = model.predict(X_test)
    pred_ref_cur = model.predict(X_cur)
    pred_drift = model.predict(X_cur_drift)
    metrics = {
        "accuracy_reference_test": accuracy_score(y_test, pred_test),
        "f1_reference_test": f1_score(y_test, pred_test),
        "accuracy_current_no_drift": accuracy_score(y_cur, pred_ref_cur),
        "accuracy_current_drifted": accuracy_score(y_cur, pred_drift),
        "f1_current_drifted": f1_score(y_cur, pred_drift),
    }
    mlflow.log_metrics(metrics)
    # cloudpickle: MLflow 3.x defaults to skops, which rejects RandomForest trees
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    # ---- drift detection (KS test + PSI) for every feature
    rows = []
    for col in X.columns:
        ks_stat, p_val = ks_2samp(X_train[col], X_cur_drift[col])
        p_i = psi(X_train[col].values, X_cur_drift[col].values)
        rows.append({"feature": col, "ks_statistic": ks_stat, "p_value": p_val,
                     "psi": p_i,
                     "drift_detected": bool(p_val < KS_ALPHA and p_i > PSI_THRESHOLD)})
    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    n_drift = int(report["drift_detected"].sum())

    # prediction drift: share of 'benign' predictions
    pred_share_ref = float(np.mean(pred_test))
    pred_share_cur = float(np.mean(pred_drift))

    mlflow.log_metrics({
        "n_features_total": len(X.columns),
        "n_features_drifted": n_drift,
        "share_features_drifted": n_drift / len(X.columns),
        "max_psi": float(report["psi"].max()),
        "prediction_share_benign_reference": pred_share_ref,
        "prediction_share_benign_current": pred_share_cur,
    })

    # ---- artifacts: csv report + plots
    csv_path = os.path.join(OUT_DIR, "drift_report.csv")
    report.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, drifted_features):
        ax.hist(X_train[col], bins=25, alpha=0.6, label="Reference")
        ax.hist(X_cur_drift[col], bins=25, alpha=0.6, label="Current (drifted)")
        ax.set_title(col)
        ax.legend()
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "feature_distributions.png")
    fig.savefig(hist_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(hist_path)

    top = report.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["psi"],
            color=["tab:red" if v else "tab:gray" for v in top["drift_detected"]])
    ax.axvline(PSI_THRESHOLD, ls="--", color="k", label=f"PSI threshold = {PSI_THRESHOLD}")
    ax.set_xlabel("PSI")
    ax.set_title("Top 10 features by PSI")
    ax.legend()
    fig.tight_layout()
    psi_path = os.path.join(OUT_DIR, "psi_top10.png")
    fig.savefig(psi_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(psi_path)

    mlflow.set_tag("drift_type", "data drift (simulated covariate shift)")
    mlflow.set_tag("drift_detected", str(n_drift > 0))

# ------------------------------------------------------------- 3. console
print("\n=== Performance ===")
for k, v in metrics.items():
    print(f"{k:30s}: {v:.4f}")
print("\n=== Drift report (top 8 by PSI) ===")
print(report.head(8).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nFeatures with drift: {n_drift} / {len(X.columns)}")
print(f"Benign-prediction share  reference={pred_share_ref:.3f}  current={pred_share_cur:.3f}")
print("\nDone. Start the UI with:  mlflow ui   ->  http://127.0.0.1:5000")
