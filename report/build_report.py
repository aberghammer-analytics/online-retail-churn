"""Build a self-contained interactive HTML report for the churn project.

Reproduces the modelling from ``modeling.ipynb`` (logistic-regression baseline +
tuned random forest), then exports every number the report needs as a single
JSON blob and injects it into ``report_template.html`` to produce a standalone
``report.html`` (no server, no CDN, no external files).

Run with:
    uv run build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------------------
# Config (mirrors modeling.ipynb)
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2
N_SPLITS = 5

DATA_DIR = Path("data")
BRONZE_PATH = DATA_DIR / "bronze.parquet"
SILVER_PATH = DATA_DIR / "silver.parquet"
TEMPLATE_PATH = Path("report/report_template.html")
OUTPUT_PATH = Path("report.html")

DROP_COLS = ["first_purchase", "last_purchase", "total_units"]
LOG_COLS = [
    "total_revenue",
    "avg_order_value",
    "avg_basket_size",
    "n_purchases",
    "last_gap_vs_median",
]

# Plain-language descriptions for every silver feature, used in the report UI.
FEATURE_DOCS = {
    "n_purchases": "Number of separate invoices (purchases) before the cutoff.",
    "median_gap": "Median number of days between consecutive purchases.",
    "total_revenue": "Total GBP spent across all pre-cutoff purchases.",
    "avg_order_value": "Mean revenue per invoice.",
    "avg_basket_size": "Mean number of units per invoice.",
    "avg_unique_items": "Mean number of distinct products per invoice.",
    "last_gap_vs_median": "Most recent purchase gap divided by the customer's median gap (a deceleration signal).",
    "recency": "Days from the last purchase to the cutoff (higher = quieter lately).",
    "tenure": "Days from the first purchase to the cutoff (relationship length).",
    "recency_vs_gap": "Recency expressed in units of the customer's typical gap.",
    "churn": "Target: 1 if no product purchase in the final 90 days, else 0.",
}

# Non-product stock codes (kept in sync with build_silver.py) for the funnel.
NON_PRODUCT = {
    "M", "POST", "DOT", "D", "C2", "C3", "S", "B",
    "BANK CHARGES", "AMAZONFEE", "CRUK", "ADJUST", "ADJUST2",
}
CUTOFF_N_DAYS = 90
MIN_PURCHASES = 2


def precision_at_k(y_true, y_scores, k):
    """Fraction of true churners among the top-k highest-scored customers."""
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = int(np.ceil(k * n)) if k <= 1 else int(k)
    top = np.argsort(y_scores)[::-1][:k]
    return float(y_true[top].mean())


def precision_lift_table(y_true, y_scores, ks=(0.1, 0.2, 0.3)):
    base = float(np.mean(y_true))
    rows = []
    for k in ks:
        p = precision_at_k(y_true, y_scores, k)
        rows.append({"k": int(k * 100), "precision": p, "lift": p / base})
    return rows


def clean_report(y_true, y_pred):
    """classification_report as a plain nested dict of floats."""
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    out = {}
    for key, val in rep.items():
        if isinstance(val, dict):
            out[key] = {k: float(v) for k, v in val.items()}
        else:
            out[key] = float(val)
    return out


def build_funnel() -> dict:
    """Row-count funnel from bronze line items down to silver customers."""
    bronze = pd.read_parquet(BRONZE_PATH)
    code = bronze["stock_code"].astype(str).str.upper().str.strip()
    products = bronze[~code.isin(NON_PRODUCT) & (bronze["quantity"] > 0)]
    cutoff = bronze["invoice_date"].max() - pd.Timedelta(days=CUTOFF_N_DAYS)
    pre_cutoff = products[products["invoice_date"] < cutoff]
    n_invoices = pre_cutoff.groupby(["invoice", "customer_id"]).ngroups
    return {
        "cutoff_date": str(cutoff.date()),
        "stages": [
            {"label": "Bronze line items", "value": int(len(bronze))},
            {"label": "Product line items (positive qty)", "value": int(len(products))},
            {"label": "Pre-cutoff invoices", "value": int(n_invoices)},
            {"label": "Silver customers (≥2 purchases)", "value": None},  # filled later
        ],
    }


def main() -> None:
    # --- Load silver, mirror notebook preprocessing -----------------------
    df = pd.read_parquet(SILVER_PATH).set_index("customer_id")
    model_feats = df.drop(columns=DROP_COLS)
    X = model_feats.drop(columns=["churn"])
    y = model_feats["churn"]
    feat_names = list(X.columns)
    NUM_COLS = [c for c in X.columns if c not in LOG_COLS]

    logger.info(f"Silver: {df.shape[0]:,} customers, churn rate {y.mean():.3f}")

    # --- EDA: skew, churn-by-median, correlation --------------------------
    skew = model_feats.drop(columns=["churn"]).skew()
    eda_features = {}
    for col in feat_names:
        med = float(model_feats[col].median())
        low = float(model_feats.loc[model_feats[col] <= med, "churn"].mean())
        high = float(model_feats.loc[model_feats[col] > med, "churn"].mean())
        eda_features[col] = {
            "skew": float(skew[col]),
            "median": med,
            "churn_below_median": low,
            "churn_above_median": high,
            "doc": FEATURE_DOCS.get(col, ""),
            "min": float(X[col].min()),
            "max": float(X[col].max()),
        }
    corr = model_feats.drop(columns=["churn"]).corr()
    high_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .loc[lambda s: s.abs() > 0.85]
    )
    eda = {
        "features": eda_features,
        "high_corr_pairs": [
            {"a": a, "b": b, "r": float(r)} for (a, b), r in high_pairs.items()
        ],
    }

    # --- Train/test split (leakage-safe, stratified) ----------------------
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    logger.info(f"Train {X_tr.shape}, test {X_te.shape}")

    # --- Logistic regression baseline -------------------------------------
    prep = ColumnTransformer(
        transformers=[("log", Pipeline([("log", FunctionTransformer(np.log1p))]), LOG_COLS)],
        remainder="passthrough",
    )
    logit = Pipeline([
        ("prep", prep),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    logit_cv = cross_val_score(logit, X_tr, y_tr, cv=cv, scoring="roc_auc")
    logit.fit(X_tr, y_tr)

    # Coefficients are in transformed-column order: LOG_COLS + remainder.
    coef_order = LOG_COLS + NUM_COLS
    coef = logit.named_steps["clf"].coef_[0]
    scaler = logit.named_steps["scale"]
    logit_coef = {name: float(c) for name, c in zip(coef_order, coef)}

    # --- Random forest (default, then tuned) ------------------------------
    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ])
    rf_cv = cross_val_score(rf, X_tr, y_tr, cv=cv, scoring="roc_auc")

    param_dist = {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [None, 6, 10, 16],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=cv, scoring="roc_auc",
        random_state=SEED, n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    rf = search.best_estimator_
    logger.info(f"Best RF params: {search.best_params_} (CV AUC {search.best_score_:.3f})")

    rf_importances = {
        name: float(v)
        for name, v in zip(feat_names, rf.named_steps["clf"].feature_importances_)
    }

    # --- Evaluate on holdout ---------------------------------------------
    logit_proba = logit.predict_proba(X_te)[:, 1]
    rf_proba = rf.predict_proba(X_te)[:, 1]

    def model_block(proba, cv_scores, extra=None):
        block = {
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "test_auc": float(roc_auc_score(y_te, proba)),
            "base_rate": float(y_te.mean()),
            "precision_at_k": precision_lift_table(y_te, proba),
            "report": clean_report(y_te, (proba >= 0.5).astype(int)),
        }
        if extra:
            block.update(extra)
        return block

    models = {
        "logit": model_block(
            logit_proba, logit_cv, {"coef": logit_coef}
        ),
        "rf": model_block(
            rf_proba, rf_cv,
            {
                "tuned_cv_auc": float(search.best_score_),
                "best_params": {k: search.best_params_[k] for k in search.best_params_},
                "importances": rf_importances,
            },
        ),
    }

    # What-if payload: reproduce the logistic pipeline in the browser.
    models["logit"]["whatif"] = {
        "order": coef_order,
        "log_cols": LOG_COLS,
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in coef],
        "intercept": float(logit.named_steps["clf"].intercept_[0]),
    }

    # RF test-set predictions for the threshold slider + top-risk table.
    test_preds = [
        {"customer_id": float(cid), "proba": float(p), "actual": int(a)}
        for cid, p, a in zip(X_te.index, rf_proba, y_te.to_numpy())
    ]

    # --- Honest out-of-fold RF scores for ALL customers (explorer) --------
    oof = cross_val_predict(rf, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    customers = []
    for i, cid in enumerate(X.index):
        row = {"customer_id": float(cid), "risk": float(oof[i]), "churn": int(y.iloc[i])}
        for col in feat_names:
            row[col] = float(X.iloc[i][col])
        customers.append(row)

    # --- Pipeline metadata ------------------------------------------------
    funnel = build_funnel()
    funnel["stages"][-1]["value"] = int(df.shape[0])
    churn_counts = y.value_counts().to_dict()

    bronze_cols = [
        ("invoice", "string", "Transaction ID (“C” prefix = cancellation)"),
        ("stock_code", "string", "Product identifier (alphanumeric)"),
        ("description", "object", "Product name"),
        ("quantity", "int64", "Units purchased (negative = return)"),
        ("invoice_datetime", "datetime64", "Timestamp of the transaction"),
        ("price", "float64", "Unit price (GBP)"),
        ("customer_id", "float64", "Unique customer identifier"),
        ("country", "object", "Country of purchase"),
        ("invoice_date", "datetime64", "Date only (derived)"),
        ("invoice_month_year", "string", "“YYYY-MM” for sorting (derived)"),
        ("line_item_total", "float64", "quantity × price (derived)"),
    ]
    silver_cols = [(c, str(df[c].dtype), FEATURE_DOCS.get(c, "")) for c in df.columns]

    pipeline = {
        "constants": {
            "cutoff_n_days": CUTOFF_N_DAYS,
            "min_purchases": MIN_PURCHASES,
            "cutoff_date": funnel["cutoff_date"],
            "non_product": sorted(NON_PRODUCT),
        },
        "funnel": funnel,
        "churn_distribution": {
            "retained": int(churn_counts.get(0, 0)),
            "churned": int(churn_counts.get(1, 0)),
        },
        "bronze_schema": [
            {"name": n, "dtype": d, "doc": doc} for n, d, doc in bronze_cols
        ],
        "silver_schema": [
            {"name": n, "dtype": d, "doc": doc} for n, d, doc in silver_cols
        ],
        "code": {
            "bronze": Path("scripts/build_bronze.py").read_text(),
            "silver": Path("scripts/build_silver.py").read_text(),
        },
    }

    report = {
        "feature_names": feat_names,
        "n_customers": int(df.shape[0]),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "n_splits": N_SPLITS,
        "drop_cols": DROP_COLS,
        "log_cols": LOG_COLS,
        "pipeline": pipeline,
        "eda": eda,
        "models": models,
        "test_predictions": test_preds,
        "customers": customers,
    }

    # --- Inject into template --------------------------------------------
    template = TEMPLATE_PATH.read_text()
    payload = json.dumps(report, separators=(",", ":"))
    html = template.replace("__REPORT_DATA__", payload)
    OUTPUT_PATH.write_text(html)

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    logger.success(
        f"Wrote {OUTPUT_PATH} ({size_mb:.2f} MB) — "
        f"RF test AUC {models['rf']['test_auc']:.3f}, "
        f"logit test AUC {models['logit']['test_auc']:.3f}"
    )


if __name__ == "__main__":
    main()
