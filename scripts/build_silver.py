"""Build the silver layer for the Online Retail II churn project.

Reads the bronze line-item table, strips out non-product transactions and
returns/cancellations, then aggregates to one row per customer with the
behavioural features used to predict churn (recency, tenure, purchase cadence,
spend, and basket characteristics).

Churn is defined relative to a recency cutoff: a customer "churns" if they made
no product purchase in the final CUTOFF_N_DAYS of the data. All features are
measured *to the cutoff*, never to the end of the data, so the label can't leak
into the features. The result is written to data/silver.parquet.

Run with:
    uv run build_silver.py
"""

from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path("data")
BRONZE_PATH = DATA_DIR / "bronze.parquet"
SILVER_PATH = DATA_DIR / "silver.parquet"

# Stock codes that aren't real products: postage, fees, bank charges, manual
# adjustments, etc. These distort spend and cadence features, so we drop them.
NON_PRODUCT = {
    "M",
    "POST",
    "DOT",
    "D",
    "C2",
    "C3",
    "S",
    "B",
    "BANK CHARGES",
    "AMAZONFEE",
    "CRUK",
    "ADJUST",
    "ADJUST2",
}

# Window at the end of the data used to define churn.
CUTOFF_N_DAYS = 90

# Customers need at least this many purchases to have a meaningful cadence
# (a purchase gap requires two purchases).
MIN_PURCHASES = 2


def build_silver() -> pd.DataFrame:
    """Aggregate bronze line items into per-customer churn features."""
    df = pd.read_parquet(BRONZE_PATH)

    cutoff_date = df["invoice_date"].max() - pd.Timedelta(days=CUTOFF_N_DAYS)
    logger.info(f"Cutoff date for recency: {cutoff_date}")

    # Keep only positive-quantity purchases of real products. Negative
    # quantities are returns/cancellations; non-product codes are fees etc.
    code = df["stock_code"].astype(str).str.upper().str.strip()
    products = df[~code.isin(NON_PRODUCT) & (df["quantity"] > 0)]

    # Anyone who bought in the final window is, by definition, not churned.
    no_churn_customers = products[products["invoice_date"] >= cutoff_date][
        "customer_id"
    ].unique()
    pre_cutoff = products[products["invoice_date"] < cutoff_date]

    logger.info(
        f"Customers active in the last {CUTOFF_N_DAYS} days: "
        f"{len(no_churn_customers):,}"
    )
    logger.info(f"Pre-cutoff line items: {pre_cutoff.shape}")

    # Roll line items up to the invoice (purchase) level so we can measure the
    # gaps *between purchases*, not between individual line items.
    invoice_level = pre_cutoff.groupby(["invoice", "customer_id"], as_index=False).agg(
        invoice_date=("invoice_date", "min"),
        invoice_revenue=("line_item_total", "sum"),
        invoice_n_items=("quantity", "sum"),
        invoice_unique_items=("stock_code", "nunique"),
    )

    invoice_level = invoice_level.sort_values(["customer_id", "invoice_date"])
    invoice_level["purchase_gap"] = (
        invoice_level.groupby("customer_id")["invoice_date"].diff().dt.days
    )

    feats = invoice_level.groupby("customer_id").agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        n_purchases=("invoice", "size"),
        median_gap=("purchase_gap", "median"),
        total_revenue=("invoice_revenue", "sum"),
        avg_order_value=("invoice_revenue", "mean"),
        total_units=("invoice_n_items", "sum"),
        avg_basket_size=("invoice_n_items", "mean"),
        avg_unique_items=("invoice_unique_items", "mean"),
    )

    # deceleration - ratio of last gap to median gap. If they're taking much longer than usual, maybe they're more likely to churn?
    last_gap = invoice_level.groupby("customer_id")["purchase_gap"].last()
    feats["last_gap_vs_median"] = last_gap / feats["median_gap"].clip(lower=1)

    # Recency + tenure, measured to the cutoff (never to end-of-data) so the
    # churn label can't leak into the features.
    feats["recency"] = (cutoff_date - feats["last_purchase"]).dt.days
    feats["tenure"] = (cutoff_date - feats["first_purchase"]).dt.days

    # Silence relative to the customer's own rhythm: how many typical purchase
    # gaps have elapsed since their last order. Floor the gap at 1 day so
    # same-day repeat buyers (median_gap == 0) stay finite rather than +inf.
    gap = feats["median_gap"].clip(lower=1)
    feats["recency_vs_gap"] = feats["recency"] / gap

    # Need at least two purchases for a cadence to exist.
    feats = feats[feats["n_purchases"] >= MIN_PURCHASES].copy()

    # Churn = no product purchase in the final window.
    feats["churn"] = (~feats.index.isin(no_churn_customers)).astype(int)

    # customer_id is the groupby index; make it a real column so it survives
    # the Parquet round-trip (we write with index=False, like bronze).
    feats = feats.reset_index()

    return feats


def main() -> None:
    feats = build_silver()

    logger.info(f"Columns and types:\n{feats.dtypes}")
    logger.info(f"Churn rate:\n{feats['churn'].value_counts(normalize=True)}")

    DATA_DIR.mkdir(exist_ok=True)
    feats.to_parquet(SILVER_PATH, index=False)
    logger.success(
        f"Wrote silver layer to {SILVER_PATH} "
        f"({feats.shape[0]:,} customers, {feats.shape[1]} columns)"
    )


if __name__ == "__main__":
    main()
