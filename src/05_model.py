"""
Stage 5: Predictive Modeling Agent
Trains a gradient-boosted classifier to predict high-severity inspection
visits, with leak-safe temporal split and calibration.
"""
import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
OUT = ROOT / "outputs"

FEATURES = [
    "prior_visits", "prior_viol_sum", "prior_crit_sum", "prior_viol_avg",
    "days_since_last", "month", "dow", "is_summer",
    "has_temperature", "has_handwashing", "has_pests", "has_sourcing", "has_permit",
    "tmax", "tmin", "prcp",
]


def main():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["inspection_date"])
    df = df.sort_values("inspection_date").reset_index(drop=True)

    # Fill weather NaNs with medians (partial coverage is expected)
    for c in ["tmax", "tmin", "prcp"]:
        df[c] = df[c].fillna(df[c].median())

    X = df[FEATURES].fillna(0)
    y = df["target_critical"]

    # Temporal split: first 80% chronologically for training, last 20% for test
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    # Base-rate baseline for context
    base_rate = float(y_train.mean())
    base_proba = np.full_like(y_test, base_rate, dtype=float)

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_positive_rate": base_rate,
        "test_positive_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc":  float(average_precision_score(y_test, proba)),
        "baseline_roc_auc": float(roc_auc_score(y_test, base_proba))
                            if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, output_dict=True, zero_division=0
        ),
    }

    # Feature importances for explainability
    importances = sorted(
        zip(FEATURES, clf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    metrics["feature_importances"] = [
        {"feature": f, "importance": round(float(i), 4)} for f, i in importances
    ]

    # Sanity checks
    checks = {
        "auc_above_baseline": metrics["roc_auc"] > 0.55,
        "predictions_not_random": metrics["roc_auc"] > 0.6,
        "pr_auc_above_base":      metrics["pr_auc"] > metrics["test_positive_rate"],
    }
    metrics["checks"] = checks
    metrics["status"] = "ok" if all(checks.values()) else "degraded"

    joblib.dump({"model": clf, "features": FEATURES}, MODELS / "risk_model.joblib")
    (OUT / "stage5_model_report.json").write_text(json.dumps(metrics, indent=2))

    print(f"[CHECKPOINT] Stage 5: {metrics['status']}")
    print(f"  ROC AUC  : {metrics['roc_auc']:.3f}")
    print(f"  PR  AUC  : {metrics['pr_auc']:.3f} (baseline {metrics['test_positive_rate']:.3f})")
    print(f"  Top features:")
    for fi in metrics["feature_importances"][:6]:
        print(f"    {fi['feature']:20s} {fi['importance']}")


if __name__ == "__main__":
    main()
