"""
Stage 6: Output / UI Structuring Agent
Takes a vendor scenario, returns:
  - compliance score (0-100)
  - required permits (from extracted rules)
  - risk breakdown with confidence
  - action recommendations
  - traceability (which rules + data drove each output)
"""
import json
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
RULES = ROOT / "rules"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

bundle = joblib.load(MODELS / "risk_model.joblib")
MODEL = bundle["model"]
FEATURES = bundle["features"]
RULE_BANK = json.loads((RULES / "extracted_rules.json").read_text())


def score_vendor(scenario: dict) -> dict:
    # Build feature row from scenario + sensible defaults
    row = {
        "prior_visits":    scenario.get("prior_visits", 0),
        "prior_viol_sum":  scenario.get("prior_violations_total", 0),
        "prior_crit_sum":  scenario.get("prior_critical_total", 0),
        "prior_viol_avg":  scenario.get("prior_viol_avg", 0),
        "days_since_last": scenario.get("days_since_last_inspection", -1),
        "month":           scenario.get("month", 7),
        "dow":             scenario.get("day_of_week", 2),
        "is_summer":       int(scenario.get("month", 7) in (6, 7, 8)),
        "has_temperature": int(scenario.get("handles_hot_cold_foods", True)),
        "has_handwashing": int(scenario.get("has_handwash_station", True) is False),
        "has_pests":       int(scenario.get("pest_history", False)),
        "has_sourcing":    int(scenario.get("unapproved_sources", False)),
        "has_permit":      int(not scenario.get("permit_current", True)),
        "tmax":            scenario.get("tmax_f", 78),
        "tmin":            scenario.get("tmin_f", 62),
        "prcp":            scenario.get("precip_in", 0),
    }
    X = pd.DataFrame([row])[FEATURES]
    p = float(MODEL.predict_proba(X)[0, 1])
    score = round((1 - p) * 100, 1)

    # Map model features to top risk contributors
    importances = dict(zip(FEATURES, MODEL.feature_importances_))
    contributors = sorted(
        [{"feature": f, "value": row[f], "importance": round(importances[f], 3)}
         for f in FEATURES if row[f] and importances[f] > 0.02],
        key=lambda x: x["importance"] * (1 + abs(x["value"])),
        reverse=True,
    )[:5]

    # Pull relevant permits / rules from rule bank
    required_permits = [
        {"rule_id": r["rule_id"], "section": r.get("section"), "summary": r["requirement_summary"],
         "source": r["source"], "confidence": r.get("confidence", "high")}
        for r in RULE_BANK
        if r["risk_type"] in {"permit", "commissary"}
    ]

    # Action recommendations driven by scenario + model contributors
    actions = []
    if not scenario.get("permit_current", True):
        actions.append({
            "priority": "critical",
            "action": "Renew mobile food vending permit before operating.",
            "grounded_in": "NYC DOH Mobile Food Vendor Regulations + NY Subpart 14-4.40",
        })
    if scenario.get("has_handwash_station", True) is False:
        actions.append({
            "priority": "critical",
            "action": "Install handwashing station with potable water before service.",
            "grounded_in": "NY Subpart 14-4.170",
        })
    if scenario.get("handles_hot_cold_foods", True):
        actions.append({
            "priority": "high",
            "action": "Maintain cold holding <41°F and hot holding >140°F; log every 2h.",
            "grounded_in": "NYC DOH — Hold Food at Proper Temperatures",
        })
    if scenario.get("pest_history", False):
        actions.append({
            "priority": "high",
            "action": "Document pest-control contract and inspection log.",
            "grounded_in": "NY Subpart 14-4.190",
        })
    if scenario.get("unapproved_sources", False):
        actions.append({
            "priority": "critical",
            "action": "Source all food from approved commissary / licensed suppliers only.",
            "grounded_in": "NY Subpart 14-4.60",
        })

    confidence = "high" if p < 0.2 or p > 0.8 else "medium" if p < 0.35 or p > 0.65 else "low"

    result = {
        "vendor": scenario.get("vendor_name", "Unknown"),
        "compliance_score": score,
        "predicted_risk_probability": round(p, 3),
        "risk_band": (
            "low" if score >= 75 else "medium" if score >= 50 else "high"
        ),
        "model_confidence": confidence,
        "top_risk_contributors": contributors,
        "required_permits_and_rules": required_permits,
        "recommended_actions": actions,
        "derived_from": {
            "data_sources": [
                "Suffolk County Restaurant Violations 2024-2025 (ArcGIS FeatureServer)",
                "NOAA NCEI Daily Summaries - Islip KISP",
                "NYC DOH Mobile Food Vendor Regulations (PDF)",
                "NY State Sanitary Code Subpart 14-4",
            ],
            "model": "GradientBoostingClassifier — ROC AUC 0.859 on temporal holdout",
            "rules_version": "rules/extracted_rules.json",
        },
    }
    return result


def run_examples():
    scenarios = [
        {
            "vendor_name": "Montauk Lobster Roll Truck",
            "permit_current": True,
            "has_handwash_station": True,
            "handles_hot_cold_foods": True,
            "unapproved_sources": False,
            "pest_history": False,
            "prior_visits": 4,
            "prior_violations_total": 5,
            "prior_critical_total": 0,
            "prior_viol_avg": 1.25,
            "days_since_last_inspection": 180,
            "month": 7, "day_of_week": 5, "tmax_f": 84, "tmin_f": 68, "precip_in": 0.0,
        },
        {
            "vendor_name": "Huntington Halal Cart (lapsed permit)",
            "permit_current": False,
            "has_handwash_station": False,
            "handles_hot_cold_foods": True,
            "unapproved_sources": True,
            "pest_history": True,
            "prior_visits": 6,
            "prior_violations_total": 42,
            "prior_critical_total": 15,
            "prior_viol_avg": 7.0,
            "days_since_last_inspection": 90,
            "month": 8, "day_of_week": 1, "tmax_f": 91, "tmin_f": 72, "precip_in": 0.2,
        },
        {
            "vendor_name": "Patchogue Coffee Cart (new vendor)",
            "permit_current": True,
            "has_handwash_station": True,
            "handles_hot_cold_foods": False,
            "unapproved_sources": False,
            "pest_history": False,
            "prior_visits": 0,
            "prior_violations_total": 0,
            "prior_critical_total": 0,
            "prior_viol_avg": 0,
            "days_since_last_inspection": -1,
            "month": 5, "day_of_week": 3, "tmax_f": 72, "tmin_f": 55, "precip_in": 0.0,
        },
    ]
    results = [score_vendor(s) for s in scenarios]

    # Final validation checkpoint
    failures = []
    for r in results:
        if not (0 <= r["compliance_score"] <= 100):
            failures.append(f"score out of range for {r['vendor']}")
        if r["predicted_risk_probability"] < 0 or r["predicted_risk_probability"] > 1:
            failures.append(f"bad probability for {r['vendor']}")
        if not r["required_permits_and_rules"]:
            failures.append(f"no rules grounded for {r['vendor']}")

    # Cross-check: "lapsed permit" scenario should score lower than "new compliant"
    lapsed = next(r for r in results if "lapsed" in r["vendor"])
    compliant = next(r for r in results if "Montauk" in r["vendor"])
    if lapsed["compliance_score"] >= compliant["compliance_score"]:
        failures.append("monotonicity violated: risky vendor scored higher than compliant")

    final = {
        "stage": "predict",
        "status": "ok" if not failures else "degraded",
        "failures": failures,
        "results": results,
    }
    (OUT / "stage6_predictions.json").write_text(json.dumps(final, indent=2))

    print(f"[FINAL CHECKPOINT] Stage 6: {final['status']}")
    for r in results:
        print(f"\n{r['vendor']}")
        print(f"  compliance_score : {r['compliance_score']}  ({r['risk_band']}, {r['model_confidence']} conf)")
        print(f"  risk probability : {r['predicted_risk_probability']}")
        print(f"  actions          : {len(r['recommended_actions'])}")
        for a in r["recommended_actions"][:3]:
            print(f"     [{a['priority']}] {a['action']}")
    if failures:
        print("\nFAILURES:", failures)


if __name__ == "__main__":
    run_examples()
