"""
Stage 1: Data Ingestion Agent
Fetches real data from NY State + NOAA. Includes fallbacks and verification.
"""
import json
import sys
from pathlib import Path
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

REPORT = {"stage": "ingest", "sources": [], "status": "pending"}


def log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def fetch_ny_food_inspections():
    """
    NY State Department of Health - Food Service Establishment Inspections
    Socrata open data. Tries several known dataset IDs.
    """
    candidates = [
        # (label, url)
        ("NYSDOH Food Service Inspections (cnih-y5dw)",
         "https://health.data.ny.gov/resource/cnih-y5dw.csv?$limit=50000"),
        ("NYSDOH Food Service Last Inspection (sdus-mrfh)",
         "https://health.data.ny.gov/resource/sdus-mrfh.csv?$limit=50000"),
        ("NYSDOH Food Establishment (gei9-3ktm)",
         "https://health.data.ny.gov/resource/gei9-3ktm.csv?$limit=50000"),
    ]
    for label, url in candidates:
        try:
            log("TRY", f"{label} -> {url}")
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 500:
                path = RAW / "ny_food_inspections.csv"
                path.write_bytes(r.content)
                df = pd.read_csv(path, low_memory=False)
                log("OK", f"{label}: {len(df)} rows, {len(df.columns)} cols")
                REPORT["sources"].append({
                    "name": "ny_food_inspections",
                    "label": label,
                    "url": url,
                    "rows": len(df),
                    "cols": list(df.columns)[:25],
                    "file": str(path),
                    "status": "ok",
                })
                return path
            else:
                log("FAIL", f"status={r.status_code} bytes={len(r.content)}")
        except Exception as e:
            log("ERR", f"{label}: {e}")
    REPORT["sources"].append({"name": "ny_food_inspections", "status": "failed"})
    return None


def fetch_noaa_weather():
    """
    NOAA NCEI Access Data Service - daily summaries from Islip/MacArthur (KISP).
    No API key needed.
    """
    url = (
        "https://www.ncei.noaa.gov/access/services/data/v1"
        "?dataset=daily-summaries"
        "&stations=USW00054790"  # Islip MacArthur Airport
        "&startDate=2023-01-01"
        "&endDate=2024-12-31"
        "&dataTypes=TMAX,TMIN,PRCP,SNOW,AWND"
        "&format=csv"
        "&units=standard"
    )
    try:
        log("TRY", f"NOAA NCEI -> {url[:80]}...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 200:
            path = RAW / "noaa_weather.csv"
            path.write_bytes(r.content)
            df = pd.read_csv(path)
            log("OK", f"NOAA: {len(df)} rows")
            REPORT["sources"].append({
                "name": "noaa_weather",
                "url": url,
                "rows": len(df),
                "cols": list(df.columns),
                "file": str(path),
                "status": "ok",
            })
            return path
        log("FAIL", f"NOAA status={r.status_code}")
    except Exception as e:
        log("ERR", f"NOAA: {e}")
    REPORT["sources"].append({"name": "noaa_weather", "status": "failed"})
    return None


def fetch_regulations():
    """
    NY State Sanitary Code Part 14 Subpart 14-4 - Mobile Food.
    Grab the HTML for RAG. Also try ecfr-style fallback.
    """
    urls = [
        ("NY Subpart 14-4",
         "https://regs.health.ny.gov/volume-a-1/content/subpart-14-4-mobile-food-service-establishments-and-pushcarts"),
        ("NY Part 14",
         "https://regs.health.ny.gov/volume-a-1/content/part-14-food-service-establishments"),
    ]
    saved = []
    for label, url in urls:
        try:
            log("TRY", f"{label} -> {url}")
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200 and len(r.text) > 1000:
                name = label.lower().replace(" ", "_") + ".html"
                path = RAW / name
                path.write_text(r.text, encoding="utf-8")
                log("OK", f"{label}: {len(r.text)} chars")
                saved.append({"label": label, "url": url, "file": str(path), "status": "ok"})
            else:
                log("FAIL", f"{label} status={r.status_code}")
                saved.append({"label": label, "url": url, "status": f"http_{r.status_code}"})
        except Exception as e:
            log("ERR", f"{label}: {e}")
            saved.append({"label": label, "url": url, "status": f"error: {e}"})
    REPORT["sources"].append({"name": "ny_regulations", "items": saved})
    return saved


def checkpoint():
    ok = [s for s in REPORT["sources"] if s.get("status") == "ok"
          or any(i.get("status") == "ok" for i in s.get("items", []))]
    REPORT["status"] = "ok" if len(ok) >= 2 else "degraded"
    REPORT["summary"] = f"{len(ok)}/{len(REPORT['sources'])} source groups succeeded"
    (ROOT / "outputs" / "stage1_ingest_report.json").parent.mkdir(exist_ok=True)
    (ROOT / "outputs" / "stage1_ingest_report.json").write_text(
        json.dumps(REPORT, indent=2, default=str)
    )
    log("CHECKPOINT", f"Stage 1 status: {REPORT['status']}")
    log("CHECKPOINT", REPORT["summary"])


if __name__ == "__main__":
    fetch_ny_food_inspections()
    fetch_noaa_weather()
    fetch_regulations()
    checkpoint()
    print("\n=== STAGE 1 REPORT ===")
    print(json.dumps(REPORT, indent=2, default=str)[:2000])
