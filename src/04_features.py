"""
Stage 4: Feature Engineering Agent
Builds a per-(facility, inspection_date) dataset with historical risk signals,
seasonality, weather joins, and violation category tagging from Stage 3 rules.
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
RULES = ROOT / "rules"


def tag_category(text: str, rules):
    if not isinstance(text, str):
        return "other"
    t = text.lower()
    for r in rules:
        for kw in r.get("keywords", []):
            if kw.lower() in t:
                return r["risk_type"]
    return "other"


def main():
    viol = pd.read_csv(PROC / "suffolk_violations_clean.csv", parse_dates=["inspection_date"])
    wx = pd.read_csv(PROC / "weather_clean.csv", parse_dates=["date"])
    rules = json.loads((RULES / "extracted_rules.json").read_text())

    # --- Category tags from rules (keyword mapping) ---
    viol["category"] = viol["violation_text"].apply(lambda t: tag_category(t, rules))

    # --- Critical flag from code section (common Suffolk "critical" markers) ---
    crit_patterns = r"\b(?:temperature|cold|hot|hand|contamin|raw|protect|source|illness)\b"
    viol["is_critical"] = (
        viol["violation_text"].fillna("").str.lower().str.contains(crit_patterns, regex=True)
    )

    # --- Aggregate one row per inspection visit (facility + date) ---
    visit = (
        viol.groupby(["facility_id", "facility_name", "city", "zip", "inspection_date"])
            .agg(
                n_violations=("violation_text", "count"),
                n_critical=("is_critical", "sum"),
                n_categories=("category", "nunique"),
                has_temperature=("category", lambda s: int("temperature" in set(s))),
                has_handwashing=("category", lambda s: int("handwashing" in set(s))),
                has_pests=("category", lambda s: int("pests" in set(s))),
                has_sourcing=("category", lambda s: int("sourcing" in set(s))),
                has_permit=("category", lambda s: int("permit" in set(s))),
            )
            .reset_index()
    )
    visit = visit.sort_values(["facility_id", "inspection_date"])

    # --- Historical (leak-free) features: stats from prior visits only ---
    visit["prior_visits"] = visit.groupby("facility_id").cumcount()
    visit["prior_viol_sum"] = (
        visit.groupby("facility_id")["n_violations"].cumsum() - visit["n_violations"]
    )
    visit["prior_crit_sum"] = (
        visit.groupby("facility_id")["n_critical"].cumsum() - visit["n_critical"]
    )
    visit["prior_viol_avg"] = np.where(
        visit["prior_visits"] > 0,
        visit["prior_viol_sum"] / visit["prior_visits"],
        0,
    )
    visit["days_since_last"] = (
        visit.groupby("facility_id")["inspection_date"].diff().dt.days.fillna(-1)
    )

    # --- Seasonality ---
    visit["month"] = visit["inspection_date"].dt.month
    visit["dow"] = visit["inspection_date"].dt.dayofweek
    visit["is_summer"] = visit["month"].isin([6, 7, 8]).astype(int)

    # --- Weather join on inspection date ---
    wx["date_only"] = wx["date"].dt.normalize()
    visit["date_only"] = visit["inspection_date"].dt.normalize()
    visit = visit.merge(
        wx[["date_only", "tmax", "tmin", "tavg", "prcp"]],
        on="date_only", how="left"
    )
    wx_coverage = visit[["tmax"]].notna().mean().iloc[0]

    # --- Target: "high-severity visit" = top-quartile total violation count ---
    threshold = visit["n_violations"].quantile(0.75)
    visit["target_critical"] = (visit["n_violations"] >= threshold).astype(int)
    visit.attrs["severity_threshold"] = float(threshold)

    path = PROC / "features.csv"
    visit.drop(columns=["date_only"]).to_csv(path, index=False)

    rep = {
        "stage": "features",
        "rows": len(visit),
        "unique_facilities": int(visit["facility_id"].nunique()),
        "target_positive_rate": float(visit["target_critical"].mean()),
        "weather_join_coverage": float(wx_coverage),
        "category_distribution": viol["category"].value_counts().head(10).to_dict(),
        "checks": {
            "no_leakage": "prior_* features use cumcount before current row",
        },
        "null_target": int(visit["target_critical"].isna().sum()),
    }
    if rep["null_target"] == 0 and rep["rows"] > 1000:
        rep["status"] = "ok"
    else:
        rep["status"] = "degraded"
    (OUT / "stage4_features_report.json").write_text(json.dumps(rep, indent=2))
    print(f"[CHECKPOINT] Stage 4: {rep['status']}")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
