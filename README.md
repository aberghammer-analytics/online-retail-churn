# Online Retail II — Customer Churn

Predict which customers are about to stop buying, using the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
dataset (UCI): ~1M invoice line items from a UK-based online gift retailer,
spanning Dec 2009 – Dec 2011.

The project takes raw Excel transactions, builds per-customer behavioural
features, trains two churn models (a logistic-regression baseline and a tuned
random forest), and packages every result into a single self-contained
interactive HTML report.

## What "churn" means here

Churn is defined relative to a **reference cutoff date** 90 days before the end
of the data: a customer **churns** if they make no purchase in the 90 days
*following* that cutoff. A customer who does purchase after the cutoff counts as
non-churn — *unless* that was their first purchase.

The 90-day window is a deliberately **adjustable parameter**, used here as a
starting point. It was checked against the median inter-purchase gap (~56 days
for repeat customers, with a large ~84-day standard deviation): no cutoff
cleanly separates lapsed customers, but 90 days sits past the typical reorder
cycle, and the parameter can be re-tuned and re-evaluated.

Two filters shape who is eligible to be scored:

- **Returns and non-standard transactions are removed** — they don't represent typical customer interactions.
- **Customers who only purchased once before the cutoff are removed** — they lack enough purchase history to be judged churned. This also drops anyone whose first purchase fell after the 90-day cutoff.

Every predictive feature is measured *up to the cutoff* — never to the end of
the data — so the churn label's time window can't leak into the features.

## Architecture

The pipeline follows a [medallion](https://www.databricks.com/glossary/medallion-architecture)
layout. Each stage reads the previous stage's Parquet file and writes its own,
so stages are independently re-runnable and easy to inspect.

```
online_retail_II.xlsx          raw source (two yearly sheets)
        │  build_bronze.py
        ▼
data/bronze.parquet            cleaned, typed line items (1 row per line item)
        │  build_silver.py
        ▼
data/silver.parquet            churn features      (1 row per customer)
        │  build_report.py
        ▼
report.html                    standalone interactive report
```

| Stage | Script | Input | Output | Does |
|-------|--------|-------|--------|------|
| **Bronze** | `scripts/build_bronze.py` | `data/online_retail_II.xlsx` | `data/bronze.parquet` | Concatenates both sheets, applies a snake_case schema, drops rows missing an invoice or `customer_id`, and adds derived columns (`invoice_date`, `line_item_total`, …). |
| **Silver** | `scripts/build_silver.py` | `data/bronze.parquet` | `data/silver.parquet` | Removes returns and non-product codes (postage, fees, adjustments), rolls line items up to invoices, then aggregates to one row per customer with recency, tenure, cadence, spend, and basket features. Adds the `churn` label. |
| **Report** | `report/build_report.py` | `data/silver.parquet` (+ bronze for the funnel) | `report.html` | Re-runs the full modelling (logit + tuned RF), computes EDA/metrics/feature importances, and injects it all as JSON into `report/report_template.html`. |

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency and
environment management. Python **3.12+** is required (see `.python-version`).

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies into a local .venv (reads pyproject.toml / uv.lock)
uv sync
```

### Get the data

The raw dataset is **not** committed (it's ~46 MB and `data/` is gitignored).
Download `online_retail_II.xlsx` from the
[UCI repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and
place it at:

```
data/online_retail_II.xlsx
```

## Running the pipeline

Run the stages in order, **from the repo root**. `uv run` executes each script
inside the project environment — no manual `activate` needed. (The scripts use
paths relative to the working directory, so always invoke them from the root.)

```bash
uv run scripts/build_bronze.py     # -> data/bronze.parquet
uv run scripts/build_silver.py     # -> data/silver.parquet
uv run report/build_report.py      # -> report.html
```

Then open `report.html` in any browser. It's fully standalone (no server, no
CDN, no external files), so you can also just double-click it or share the file
directly.

Each script logs its progress (row counts, churn rate, cutoff date, model AUCs)
via [`loguru`](https://github.com/Delgan/loguru), so you can sanity-check the
output as it runs.

## The report

`report.html` is the main deliverable. It embeds every number from the run as a
JSON blob and renders an interactive walkthrough of the project:

- **Pipeline & schema** — the bronze→silver row-count funnel, column dictionaries, and the actual source of `build_bronze.py` / `build_silver.py`.
- **EDA** — per-feature skew, churn rate above vs. below the median, and highly-correlated feature pairs.
- **Models** — cross-validated and holdout AUC for both models, precision@k lift tables, classification reports, logistic coefficients, and RF feature importances.
- **What-if calculator** — reproduces the logistic pipeline's math in the browser so you can adjust a customer's features and watch the predicted churn probability move.
- **Risk explorer** — honest out-of-fold RF risk scores for every customer, with a threshold slider and a top-risk table.

## Project layout

```
.
├── scripts/
│   ├── build_bronze.py             # stage 1: raw Excel -> cleaned line items
│   └── build_silver.py             # stage 2: line items -> per-customer churn features
├── report/
│   ├── build_report.py             # stage 3: model + render the HTML report
│   └── report_template.html        # HTML/JS shell with a __REPORT_DATA__ placeholder
├── report.html                     # generated, self-contained report (the deliverable)
├── modeling.ipynb                  # exploratory modelling notebook
├── Subsalt Takehome Assignment.pdf # the assignment write-up
├── pyproject.toml                  # dependencies (managed by uv)
├── uv.lock                         # pinned, reproducible dependency versions
└── data/                           # gitignored — holds source xlsx + parquet layers
```

The **`Subsalt Takehome Assignment.pdf`** at the repo root is the write-up of
the assignment this project responds to.

> **Note:** `modeling.ipynb` is the exploratory scratchpad where the modelling
> was worked out; `report/build_report.py` is the cleaned-up, reproducible
> script that produces the final report.

## Modelling notes

- **Split:** 80/20 stratified train/test, with 5-fold stratified CV on the training set. `SEED = 42` throughout for reproducibility.
- **Logistic baseline:** log-transforms skewed spend/count features, standard-scales, then fits logistic regression — interpretable, and the basis for the in-browser what-if calculator.
- **Random forest:** tuned with `RandomizedSearchCV` over depth, leaf size, and tree count, optimising ROC AUC.
- **Evaluation:** ROC AUC plus **precision@k** lift — the practical question for a retention campaign is "if we contact the top 10/20/30% of riskiest customers, how many real churners do we catch versus random?"
