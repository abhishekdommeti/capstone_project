# Zepto Data & AI Platform — Capstone Project

One repository, three connected modules built for Zepto's analytics guild:

| Module | Folder | Marks | What it does |
|---|---|---|---|
| 1 — Data Pipeline | [`/data_pipeline`](./data_pipeline) | 25 | Scrapes book-catalog data, cleans it, converts currency, loads it into a normalized SQLite database, and queries it with SQL and pandas. |
| 2 — Analytics Pipeline | [`/analytics`](./analytics) | 50 | Profiles, cleans and visualizes the Titanic dataset, then builds, tunes and compares classification and regression models on it. |
| 3 — Support Assistant | [`/support_assistant`](./support_assistant) | 25 | A small RAG service answering questions about Zepto's own delivery/returns/membership policies, via LangGraph, ChromaDB and FastAPI. |

Each module has its own `README.md` with full details; this file covers setup, how to run all three end to end, and a short summary of the design decisions behind each.

---

## Setup

Each module ships its **own `requirements.txt`**, installed into its own virtual environment, since the three modules use unrelated and occasionally conflicting dependencies (e.g. Module 3's `chromadb`/`sentence-transformers` stack vs Module 2's `scikit-learn`/`imbalanced-learn` stack).

```bash
git clone https://github.com/abhishekdommeti/capstone_project.git
cd capstone_project
```

For each module, from the repo root:

```bash
cd <module_folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

Repeat for `data_pipeline`, `analytics`, and `support_assistant` (three separate environments, or reuse one venv and reinstall between modules if you'd rather not keep three).

---

## How to run each module

### 1. Data Pipeline (`/data_pipeline`)

> Adjust the exact filenames below to match what's actually committed in this folder if they differ.

```bash
cd data_pipeline
pip install -r requirements.txt
python scrape_and_load.py        # or: run the notebook top to bottom
```

This scrapes book listings from books.toscrape.com, cleans and type-converts the fields (`price_gbp`, `rating`, `in_stock`, `price_inr`), builds the two-table SQLite schema (`categories` ↔ `books`), and runs the required SQL queries. The database file (or the script that regenerates it) and the query outputs are committed in this folder — see `data_pipeline/README.md` for the exact commands and the fixed conversion rate used (1 GBP = 105.50 INR).

### 2. Analytics Pipeline (`/analytics`)

```bash
cd analytics
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb
python reload_and_predict.py     # optional: sanity-check the saved model
```

`01_eda.ipynb` loads the Titanic dataset once (via `sns.load_dataset`, cached locally afterwards), profiles and cleans it, and saves `titanic.csv`. `02_modeling.ipynb` reads that same CSV, trains/tunes/evaluates three classifiers and a regression model, and saves the best pipeline to `models/best_pipeline.joblib`. See `analytics/README.md` for the full results table and written interpretations.

### 3. Support Assistant (`/support_assistant`)

```bash
cd support_assistant
pip install -r requirements.txt
cp .env.example .env             # defaults to MOCK_LLM=1 — no API key needed
python retrieval.py              # builds the ChromaDB index from docs/
uvicorn main:app --host 0.0.0.0 --port 7860
```

Then, in another terminal:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee if my order is under INR 149?"}'
```

Or run it in Docker instead of locally:
```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

See `support_assistant/README.md` for the full architecture write-up and example call transcripts.

---

## Design decisions — summary

**Module 1 — Data Pipeline.** Uses books.toscrape.com as a stand-in catalog (public, scraping-legal, no login/API key), since the pipeline mechanics — scrape → clean → convert → store → query — are identical regardless of product category. Currency conversion uses a fixed, project-defined rate (1 GBP = 105.50 INR) rather than a live API, so the result is fully reproducible without network access at grading time. The schema is a normalized two-table `categories` ↔ `books` design with a foreign key, so category-level queries (e.g. top-rated books per category) require a real `JOIN` rather than denormalized lookups.

**Module 2 — Analytics Pipeline.** The raw Titanic dataset is loaded exactly once and cached as `titanic.csv`, so every later step — EDA, modeling, tuning, the regression side-task — works from one consistent, offline-reproducible source. Missing values are handled by an explicit percentage-threshold rule (drop rows / impute / drop column) rather than a blanket strategy, so each decision is justified by the actual measured missingness rather than convention. All preprocessing (imputation, encoding, scaling) is fit inside a `scikit-learn` `Pipeline`/`ColumnTransformer` on the training split only, with an explicit leakage audit, so nothing about the test set influences training. The final deployed model is chosen by test-set F1 (AUC as tie-breaker) among Logistic Regression, Decision Tree and a tuned Random Forest, and the full pipeline — not just the bare classifier — is what gets saved with `joblib`, so it can be reloaded and run directly on raw, unpreprocessed input.

**Module 3 — Support Assistant.** Every LLM call is gated behind a single `MOCK_LLM` toggle, defaulting to a fully offline, deterministic mock mode — this is what's graded, and it needs no API key or network access beyond the one-time embedding-model download. Retrieval, by contrast, always runs for real in both modes: it's local (sentence-transformers + ChromaDB), needs no API key, so mocking it would add complexity for no reason. The graph is a `LangGraph` `StateGraph` with a keyword-based intent classifier routing to either a retrieval-grounded answer or a fixed fallback string, and every final answer is validated against a Pydantic schema (`answer`/`sources`/`confidence`) before being returned through a FastAPI `POST /ask` endpoint.

---

## Repository structure

```
capstone_project/
├── README.md                  # this file
├── data_pipeline/              # Module 1
├── analytics/                  # Module 2
└── support_assistant/          # Module 3
```

Commit history includes a feature branch (created, committed to at least twice, and merged back into `main`) as required by the submission guidelines — visible via `git log --graph --all`.
