#!/usr/bin/env python3

import os
import polars as pl

def explore_dataset(dataset_name: str, df: pl.DataFrame) -> None:
    """
    Prints a preview of the dataset, including its schema, head,
    descriptive statistics for numeric columns, and null counts.
    """
    print(f"\n===== EXPLORATION: {dataset_name} =====")

    print("\n-- Schema --")
    print(df.schema)

    print("\n-- Head (first 5 rows) --")
    print(df.head(5))

    numeric_cols = [
        col for col, dtype in df.schema.items()
        if dtype in (pl.Int64, pl.Float64, pl.Int32, pl.Float32)
    ]
    if numeric_cols:
        stats_df = df.select(
            [pl.col(c).mean().alias(f"{c}_mean") for c in numeric_cols]
            + [pl.col(c).std().alias(f"{c}_std") for c in numeric_cols]
            + [pl.col(c).min().alias(f"{c}_min") for c in numeric_cols]
            + [pl.col(c).max().alias(f"{c}_max") for c in numeric_cols]
            + [pl.col(c).median().alias(f"{c}_median") for c in numeric_cols]
        )
        print("\n-- Descriptive Statistics (Numeric Columns) --")
        print(stats_df)

    null_counts = df.select([
        pl.col(c).null_count().alias(f"{c}_nulls") for c in df.columns
    ])
    print("\n-- Null Counts --")
    print(null_counts)

    print(f"===== END EXPLORATION: {dataset_name} =====\n")


def clean_data(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    numeric_cols: list[str] = None,
    fill_null_defaults: dict[str, float] = None,
    drop_null_in: list[str] = None,
    enforce_positive: bool = False
) -> pl.DataFrame:
    """
    Converts a specified timestamp column, casts numeric columns,
    fills or drops nulls, removes duplicates, and optionally filters out
    non-positive values in the specified numeric columns.
    """
    if numeric_cols is None:
        numeric_cols = []
    if fill_null_defaults is None:
        fill_null_defaults = {}
    if drop_null_in is None:
        drop_null_in = []

    if timestamp_col in df.columns and df.schema.get(timestamp_col) == pl.Utf8:
        df = df.with_columns(
            pl.col(timestamp_col)
            .str.strptime(pl.Datetime, fmt="%Y-%m-%d %H:%M:%S", strict=False)
            .alias(timestamp_col)
        )

    for col_name in numeric_cols:
        if col_name in df.columns and df.schema.get(col_name) == pl.Utf8:
            df = df.with_columns(pl.col(col_name).cast(pl.Float64))

    for col_name, default_val in fill_null_defaults.items():
        if col_name in df.columns:
            df = df.with_columns(pl.col(col_name).fill_null(default_val))

    existing_critical_cols = [c for c in drop_null_in if c in df.columns]
    if existing_critical_cols:
        df = df.drop_nulls(subset=existing_critical_cols)

    df = df.unique()

    if enforce_positive and numeric_cols:
        existing_numeric_cols = [c for c in numeric_cols if c in df.columns]
        if existing_numeric_cols:
            cond = pl.col(existing_numeric_cols[0]) > 0
            for c in existing_numeric_cols[1:]:
                cond &= pl.col(c) > 0
            df = df.filter(cond)

    return df


def main():
    """
    Reads the raw data from the 'data' folder, explores it,
    renames columns as needed, cleans the datasets, and writes
    the cleaned outputs back to the same 'data' folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_bbo_path = os.path.join(script_dir, "data", "raw_bbo_data.parquet")
    raw_trade_path = os.path.join(script_dir, "data", "raw_trade_data.parquet")

    print("PHASE 1: Reading and exploring raw BBO data...")
    bbo_df_raw = pl.read_parquet(raw_bbo_path)
    explore_dataset("raw_bbo_data", bbo_df_raw)

    print("PHASE 1: Reading and exploring raw trade data...")
    trade_df_raw = pl.read_parquet(raw_trade_path)
    explore_dataset("raw_trade_data", trade_df_raw)

    bbo_df_renamed = bbo_df_raw.rename({
        "index": "timestamp",
        "bid-price": "bid",
        "ask-price": "ask",
        "bid-volume": "bid_volume",
        "ask-volume": "ask_volume"
    })

    print("PHASE 2: Cleaning BBO data...")
    bbo_df_clean = clean_data(
        df=bbo_df_renamed,
        timestamp_col="timestamp",
        numeric_cols=["bid", "bid_volume", "ask", "ask_volume"],
        fill_null_defaults={},
        drop_null_in=["timestamp", "bid", "ask"],
        enforce_positive=True
    )

    trade_df_renamed = trade_df_raw.rename({
        "index": "timestamp",
        "trade-price": "price",
        "trade-volume": "volume"
    })

    print("PHASE 2: Cleaning trade data...")
    trade_df_clean = clean_data(
        df=trade_df_renamed,
        timestamp_col="timestamp",
        numeric_cols=["price", "volume"],
        fill_null_defaults={"volume": 0},
        drop_null_in=["timestamp", "price"],
        enforce_positive=True
    )

    cleaned_bbo_path = os.path.join(script_dir, "data", "cleaned_bbo_data.parquet")
    cleaned_trade_path = os.path.join(script_dir, "data", "cleaned_trade_data.parquet")

    bbo_df_clean.write_parquet(cleaned_bbo_path)
    trade_df_clean.write_parquet(cleaned_trade_path)

    print(f"Cleaning completed. Cleaned BBO data written to: {cleaned_bbo_path}")
    print(f"Cleaning completed. Cleaned trade data written to: {cleaned_trade_path}")


if __name__ == "__main__":
    main()
