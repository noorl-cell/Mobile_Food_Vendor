"""
Stage 2: Cleaning & Normalization Agent
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)
REPORT = {"stage": "clean", "inputs": {}, "outputs": {}, "checks": {}}


def clean_suffolk():
    df = pd.read_csv(RAW / "suffolk_restaurant_violations.csv", low_memory=False)
    REPORT["inputs"]["suffolk_raw_rows"] = len(df)

    # Unify duplicate column variants (the API returned both CamelCase and snake_case).
    def pick(row, *names):
        for n in names:
            v = row.get(n)
            if pd.notna(v) and str(v).strip():
                return v
        return None

    def coalesce(col_a, col_b):
        a = df[col_a] if col_a in df.columns else pd.Series([None] * len(df))
        b = df[col_b] if col_b in df.columns else pd.Series([None] * len(df))
        return a.where(a.notna() & (a.astype(str).str.len() > 0), b)

    out = pd.DataFrame({
        "facility_id":       coalesce("FacilityID", "Facility_ID"),
        "facility_name":     coalesce("FacilityName", "FACILITY_NAME"),
        "address":           df.get("Address"),
        "city":              df.get("City"),
        "zip":               df.get("Zip"),
        "inspection_date_ms":coalesce("InspectionDate", "Inspection_Date"),
        "code_section":      coalesce("SCSanitaryCodeSection", "SC_Sanitary_Code_Section"),
        "violation_text":    coalesce("ViolationText", "Violation_Text"),
        "source_year":       df.get("source_year"),
    })

    # ArcGIS epoch ms -> datetime
    out["inspection_date"] = pd.to_datetime(
        pd.to_numeric(out["inspection_date_ms"], errors="coerce"),
        unit="ms", errors="coerce"
    )
    out = out.drop(columns=["inspection_date_ms"])

    # Normalize strings
    for c in ["facility_name", "address", "city", "code_section", "violation_text"]:
        out[c] = out[c].astype(str).str.strip().replace({"nan": None, "None": None})
    out["city"] = out["city"].str.upper()
    out["zip"] = out["zip"].astype(str).str.extract(r"(\d{5})")[0]

    # Drop rows missing critical fields
    before = len(out)
    out = out.dropna(subset=["facility_id", "inspection_date"])
    REPORT["checks"]["suffolk_dropped_missing_critical"] = before - len(out)

    # Dedupe on identity
    before = len(out)
    out = out.drop_duplicates(
        subset=["facility_id", "inspection_date", "code_section", "violation_text"]
    )
    REPORT["checks"]["suffolk_dedup_removed"] = before - len(out)

    path = PROC / "suffolk_violations_clean.csv"
    out.to_csv(path, index=False)
    REPORT["outputs"]["suffolk_clean"] = {
        "file": str(path),
        "rows": len(out),
        "date_min": str(out["inspection_date"].min()),
        "date_max": str(out["inspection_date"].max()),
        "unique_facilities": int(out["facility_id"].nunique()),
        "top_cities": out["city"].value_counts().head(10).to_dict(),
    }
    return out


def clean_weather():
    df = pd.read_csv(RAW / "noaa_weather.csv")
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    for c in ["tmax", "tmin", "prcp", "snow", "awnd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tavg"] = (df["tmax"] + df["tmin"]) / 2
    path = PROC / "weather_clean.csv"
    df.to_csv(path, index=False)
    REPORT["outputs"]["weather_clean"] = {
        "file": str(path), "rows": len(df),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }
    return df


def clean_nysdoh():
    df = pd.read_csv(RAW / "ny_food_inspections.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # Nassau is on Long Island — keep as mobile-vendor context reference
    nassau = df[df["county"].str.upper() == "NASSAU"].copy()
    path = PROC / "nysdoh_nassau_clean.csv"
    nassau.to_csv(path, index=False)
    REPORT["outputs"]["nysdoh_nassau"] = {
        "file": str(path), "rows": len(nassau),
        "critical_violations_sum": int(nassau["total_critical_violations"].fillna(0).sum()),
    }
    return nassau


def checkpoint():
    failures = []
    s = REPORT["outputs"].get("suffolk_clean", {})
    if s.get("rows", 0) < 10_000:
        failures.append("suffolk row count low")
    w = REPORT["outputs"].get("weather_clean", {})
    if w.get("rows", 0) < 100:
        failures.append("weather row count low")
    REPORT["status"] = "ok" if not failures else "degraded"
    REPORT["failures"] = failures
    (ROOT / "outputs" / "stage2_clean_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str)
    )
    print(f"[CHECKPOINT] Stage 2: {REPORT['status']}")
    print(json.dumps(REPORT["outputs"], indent=2, default=str))


if __name__ == "__main__":
    clean_suffolk()
    clean_weather()
    clean_nysdoh()
    checkpoint()
