"""Build the bronze layer for the Online Retail II churn project.

Reads the two raw sheets from the source Excel workbook, concatenates them,
renames columns to a tidy snake_case schema, drops rows with no invoice or no
customer_id, and adds a few convenience columns. The result is written to
data/bronze.parquet as the starting point for downstream feature work.

Run with:
    uv run build_bronze.py
"""

from pathlib import Path

import pandas as pd
from loguru import logger

DATA_DIR = Path("data")
SOURCE_PATH = DATA_DIR / "online_retail_II.xlsx"
BRONZE_PATH = DATA_DIR / "bronze.parquet"

SHEETS = ["Year 2009-2010", "Year 2010-2011"]

COLUMNS = [
    "invoice",
    "stock_code",
    "description",
    "quantity",
    "invoice_datetime",
    "price",
    "customer_id",
    "country",
]


def build_bronze() -> pd.DataFrame:
    """Load, combine, clean, and enrich the raw transactions."""
    frames = [pd.read_excel(SOURCE_PATH, sheet_name=sheet) for sheet in SHEETS]
    df = pd.concat(frames, ignore_index=True)
    df.columns = COLUMNS

    # invoice and stock_code are identifiers, not numbers: invoices include
    # "C"-prefixed cancellations and stock codes include values like "85123A".
    # Excel leaves these as mixed-type object columns, which pyarrow can't
    # serialize, so cast them to string explicitly.
    df["invoice"] = df["invoice"].astype("string")
    df["stock_code"] = df["stock_code"].astype("string")

    logger.info(f"Dataframe shape: {df.shape}")
    df = df[df["invoice"].notna()].copy()
    logger.info(f"Shape after dropping rows with missing invoice: {df.shape}")
    df = df[df["customer_id"].notna()].copy()
    logger.info(f"Shape after dropping rows with missing customer_id: {df.shape}")

    df["invoice_date"] = df["invoice_datetime"].dt.normalize()
    # Period dtype can't be written to Parquet (pyarrow has no Period type),
    # so store the month-year as a sortable "YYYY-MM" string.
    df["invoice_month_year"] = df["invoice_datetime"].dt.strftime("%Y-%m")

    df["line_item_total"] = df["quantity"] * df["price"]

    return df


def main() -> None:
    df = build_bronze()

    logger.info(f"Columns and types:\n{df.dtypes}")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(BRONZE_PATH, index=False)
    logger.success(f"Wrote bronze layer to {BRONZE_PATH} ({df.shape[0]:,} rows)")


if __name__ == "__main__":
    main()
