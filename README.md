# Mobile Food Vendor Compliance Assistant

An agentic AI system that ingests real inspection data, extracts regulatory rules, predicts inspection risk, tailors a permit checklist, and answers compliance questions for mobile food vendors operating in **Suffolk County, NY**.

Built as a **6-stage pipeline with validation checkpoints** plus a Streamlit UI, a RAG chatbot, and an auto-generated technical report.

---

## What it does

- 🔍 **Search** 107k real Suffolk County restaurant violation records (2024–2025)
- 📊 **Profile** any facility with a model-scored compliance rating
- ⚖️ **Score what-if scenarios** — permit lapsed? no handwash station? pest history? — and get prioritized actions with rule citations
- 📋 **Permit checklist** — dynamically filters ~10 NY / Suffolk / town permits to what your operation actually needs, with fees and legal authority
- 📜 **Regulations browser** — 21 rules grounded in source text (NYC DOH PDF + NY Subpart 14-4)
- 💬 **RAG chatbot** — ask questions in plain English, get cited answers (Claude Haiku 4.5 when API key is set, graceful retrieval-only fallback otherwise)
- 🧠 **Model metrics** — ROC AUC, PR AUC, feature importances, confusion matrix
- 📄 **Auto-generated technical report** — 16+ page DOCX with live numbers pulled from the current pipeline outputs

---

## Architecture

```
┌───────────┐   ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
│ 1. Ingest │ → │ 2. Clean │ → │ 3. Rules   │ → │ 4. Feats  │ → │ 5. Model │ → │ 6. Score │
│ Suffolk   │   │ normalize│   │ TF-IDF RAG │   │ leak-safe │   │   GBC    │   │  + trace │
│ NOAA      │   │ dedupe   │   │ grounded   │   │ + weather │   │ 0.86 AUC │   │          │
│ NYC PDF   │   │          │   │            │   │           │   │          │   │          │
└───────────┘   └──────────┘   └────────────┘   └───────────┘   └──────────┘   └──────────┘
      │              │                │                │              │             │
      ▼              ▼                ▼                ▼              ▼             ▼
   data/raw    data/processed      rules/        data/processed    models/      outputs/
                                                                                      │
                         ┌────────────────────────┬───────────────────────────────────┘
                         ▼                        ▼
                 ┌──────────────┐         ┌───────────────┐
                 │ 7. RAG Chat  │         │ build_report  │
                 │ 07_rag.py    │         │ 16-page DOCX  │
                 └──────┬───────┘         └───────────────┘
                        │
                        ▼
                  ┌───────────┐
                  │  app.py   │  (7-tab Streamlit UI)
                  └───────────┘
```

Every stage writes a JSON report to `outputs/stage{N}_*.json` with a `status` field (`ok` / `degraded`) and failure details, so anyone on the team can audit what happened on any run.

---

## Data sources (all real, all public)

| Source | What | Rows |
|---|---|---|
| [Suffolk County Open Data — Restaurant Violations](https://opendata.suffolkcountyny.gov/) (ArcGIS FeatureServer) | 2024–2025 violations for Suffolk County | 107,843 |
| [NOAA NCEI Access Data Service](https://www.ncei.noaa.gov/access/services/data/v1) | Daily weather at Islip KISP | 729 |
| [NY Health Data Portal](https://health.data.ny.gov/) (Socrata) | NY State food inspections (Nassau context) | 21,669 |
| [NYC DOH Mobile Food Vendor Regulations](https://www.nyc.gov/assets/doh/downloads/pdf/rii/regulations-for-mobile-food-vendors.pdf) | Regulation PDF for RAG | 36 pages |
| NY State Sanitary Code Subpart 14-4 | Seeded section titles | 11 rules |
| Permit bank | Hand-curated `rules/required_permits.json` | 10 permits |

No API keys required for data ingestion. The optional Anthropic API key only unlocks the conversational LLM tier of the chatbot.

---

## Project layout

```
Project V2/
├── app.py                          # Streamlit UI (7 tabs)
├── requirements.txt
├── README.md
├── .env                            # ANTHROPIC_API_KEY=...  (never commit)
├── .gitignore                      # add data/, models/, outputs/, .env, .venv/
├── src/
│   ├── 01_ingest.py                # Stage 1 — fetch real data
│   ├── 02_clean.py                 # Stage 2 — normalize + dedupe
│   ├── 03_extract_rules.py         # Stage 3 — TF-IDF rule extraction
│   ├── 04_features.py              # Stage 4 — leak-safe features + weather join
│   ├── 05_model.py                 # Stage 5 — train GradientBoostingClassifier
│   ├── 06_predict.py               # Stage 6 — scenario scoring + traceability
│   ├── 07_rag.py                   # RAG chatbot (TF-IDF retrieval + Claude)
│   └── build_report.py             # Auto-generate 16+ page DOCX report
├── data/
│   ├── raw/                        # Downloaded source files
│   └── processed/                  # Cleaned CSVs used by the app
├── rules/
│   ├── extracted_rules.json        # Output of Stage 3 (21 rules)
│   └── required_permits.json       # Hand-curated permit bank (10 permits)
├── models/
│   └── risk_model.joblib           # Trained Gradient Boosting classifier
└── outputs/
    ├── stage1_ingest_report.json
    ├── stage2_clean_report.json
    ├── stage3_rules_report.json
    ├── stage4_features_report.json
    ├── stage5_model_report.json
    ├── stage6_predictions.json
    └── Mobile_Food_Vendor_Compliance_Report.docx
```

---

## Setup

### 1. Requirements

- **Python 3.11+** (developed on 3.14)
- Internet access on first run (ingestion pulls ~10 MB from Suffolk / NOAA / NYC)

### 2. Create a virtualenv and install deps

```bash
cd "Project V2"
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

This installs everything needed: pandas, scikit-learn, requests, pypdf, streamlit, anthropic, python-dotenv, python-docx.

### 3. (Optional) Add your Anthropic API key for the LLM chatbot

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...your-key...
```

**Never commit `.env`.** The app calls `load_dotenv(".env", override=True)` at startup so the file is picked up automatically. Without a key, the chatbot still works — it falls back to retrieval-only mode and shows the top matching passages with source labels.

### 4. Recommended `.gitignore`

```
.env
.venv/
__pycache__/
*.pyc
data/raw/
data/processed/
models/*.joblib
outputs/stage*_report.json
outputs/Mobile_Food_Vendor_Compliance_Report.docx
```

---

## Running

### Full pipeline (first run or to refresh data)

```bash
python src/01_ingest.py
python src/02_clean.py
python src/03_extract_rules.py
python src/04_features.py
python src/05_model.py
python src/06_predict.py
```

Each script prints a `[CHECKPOINT]` line and writes a report to `outputs/`. If any stage shows `status: degraded`, open the report JSON for the failure list.

### Launch the UI

```bash
python -m streamlit run app.py
```

Opens at `http://localhost:8501`. **Seven tabs:**

1. **💬 Ask the Assistant** — RAG chatbot with inline `[1]` `[2]` citations + expandable sources panel
2. **🔍 Search Vendors** — full-text search across 107k violations, filter by city + year
3. **📊 Vendor Profile** — pick a facility, see history + auto-scored compliance rating
4. **⚖️ Score a Scenario** — what-if form → score, actions, contributors, permit table, traceability
5. **📋 Required Permits** — operation-profile checkboxes → filtered permit checklist + fee estimate
6. **📜 Regulations** — searchable rule bank
7. **🧠 Model Info** — ROC / PR metrics, feature-importance bar chart, confusion matrix

### Generate the technical report

After the pipeline has been run at least once:

```bash
python src/build_report.py
```

Produces `outputs/Mobile_Food_Vendor_Compliance_Report.docx` — **16+ pages, 8 tables, 18 references**, with all numbers (row counts, AUC, feature importances, scenario outputs) loaded live from the current stage reports.

---

## Model details

**Target:** top-quartile (75th percentile) violation count per inspection visit — "high-severity visit." Positive-class rate ≈ 29%.

**Features (16):** prior visit stats (leak-free cumulative), days since last inspection, month / day-of-week / summer flag, rule-extracted category flags (temperature, handwashing, pests, sourcing, permit), weather (tmax / tmin / prcp).

**Algorithm:** `sklearn.ensemble.GradientBoostingClassifier` — 200 estimators, max_depth 4, learning_rate 0.05.

**Split:** chronological 80 / 20. No future leakage into training.

**Performance on temporal holdout:**

| Metric | Value |
|---|---|
| ROC AUC | **0.859** |
| PR AUC | **0.673** |
| Baseline PR AUC (class rate) | 0.273 |
| Lift over baseline | **2.5×** |

**Top features:** `has_temperature` (36%), `has_handwashing` (18%), `prior_viol_avg` (15%), `has_permit` (13%) — all domain-sensible.

---

## RAG chatbot details

**Module:** `src/07_rag.py`

**Knowledge base (51 passages):**
- 21 extracted regulations (NYC DOH PDF + NY Subpart 14-4 sections)
- 10 permit entries from `rules/required_permits.json`
- 20 most-common real Suffolk violation texts

**Retrieval:** TF-IDF (unigram + bigram) + cosine similarity, top-k = 5.

**Generation:**
- **🟢 LLM mode** — if `ANTHROPIC_API_KEY` is set, the retrieved passages are sent as context to `claude-haiku-4-5-20251001` with a system prompt forcing answers from context only, with inline `[1]` `[2]` `[3]` citations.
- **🟡 Retrieval-only mode** — fallback when no key is set; returns the top passages with sources directly.

The mode indicator is visible in the chat tab caption.

---

## Known limitations

- **Weather overlap** — NOAA data currently covers 2023–2024 but violations are 2024–2025, so only 2024 visits get weather-joined (~50% coverage). Widen the NOAA date range in `src/01_ingest.py::fetch_noaa_weather` to fix.
- **"Critical" severity** — the Suffolk source data doesn't mark violations as critical / non-critical, so the target is proxied as top-quartile violation count per visit.
- **Mixed populations** — Suffolk's violation data mixes fixed restaurants and mobile units. For a production mobile-only deployment, filter on permit type in Stage 1.
- **Rule mix** — Stage 3 extracts 10 rules from the NYC PDF (grounded in source) plus 11 seeded Subpart 14-4 section titles (grounded in NY State law but not re-scraped).
- **Permit bank** — `rules/required_permits.json` is hand-curated from public NY / Suffolk / town sources. Verify fees with the issuing agency before publishing externally.
- **Local-only** — no auth, no database. Single-user Streamlit. For multi-user deployment wrap with `streamlit-authenticator` or put behind a reverse proxy.

---

## How to extend

| You want to… | Touch these files |
|---|---|
| Add a new data source | `src/01_ingest.py` — add a `fetch_*` function, then re-run the pipeline |
| Add a new permit type | `rules/required_permits.json` — the UI picks it up automatically |
| Add a new rule keyword | `src/03_extract_rules.py` — add to `CATEGORIES` |
| Swap the model | `src/05_model.py` — change the estimator, keep the `FEATURES` list |
| Add a chatbot data source | `src/07_rag.py` — add passages to `build_knowledge_base()` |
| Add a UI tab | `app.py` — add to the `st.tabs(...)` call and write a `with tabN:` block |
| Regenerate the report | `python src/build_report.py` — pulls live numbers from stage reports |

---

## Team conventions

- **Don't commit:** `.env`, `data/raw/`, `data/processed/`, `models/`, `outputs/*.json`, `outputs/*.docx`
- **Do commit:** source in `src/`, `rules/*.json`, `app.py`, `requirements.txt`, `README.md`
- **Reproducibility:** running `src/01_ingest.py` → `src/06_predict.py` on a clean clone should rebuild everything. If it doesn't, that's a bug — file it.
- **Stage reports** in `outputs/stage*_report.json` are the source of truth for "did it work." Link them in PRs.
- **Secrets:** API keys only in `.env` or the OS env. Never paste keys into chat, commits, or issue tracker.

---

## Quick troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Chatbot stays in 🟡 retrieval-only mode even with `.env` | Environment already has an empty `ANTHROPIC_API_KEY` | `load_dotenv(".env", override=True)` — already set in `app.py` |
| `HTTP 404` on NY regs fetch in Stage 1 | State URL changed | The pipeline uses the NYC PDF as primary; HTML fetch is a secondary fallback |
| `KeyError` on a stage report field | Stale pipeline output | Re-run the relevant stage — each run overwrites its report |
| Streamlit won't start — port in use | Previous instance still running | `python -m streamlit run app.py --server.port 8502` |
| Suffolk ingest returns only 1000 rows | Pagination loop truncated | Check `src/01_ingest.py` — the service paginates at 1000/request, the loop should continue while `len(feats) == step` |
| `UnicodeEncodeError` when printing in Windows | Non-ASCII chars in console output | Set `PYTHONIOENCODING=utf-8` or avoid Unicode in print statements |
| Report generation fails | `python-docx` not installed | `pip install -r requirements.txt` |

---

## License

Internal team use. Data is public domain / public record; code is proprietary to the team. The Anthropic API is used under your own API agreement.
