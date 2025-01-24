import polars as pl
import pandas as pd
import os

def final_data_bbo(path: str) -> pl.DataFrame:
    """
    Insert additional columns to the BBO data.

    Args:
        path (str): Path to where raw_bbo_data.parquet is stored.

    Returns:
        Polars.DataFrame: Return a Polars DataFrame with additional columns.
    """
    
    data_bbo = pd.read_parquet("data/cleaned_bbo_data.parquet")
    data_bbo['mid_price'] = (data_bbo['bid'] + data_bbo['ask']) / 2
    data_bbo['simple_return'] = data_bbo.groupby('Stock')['mid_price'].pct_change()
    data_bbo = pl.from_pandas(data_bbo)
    
    data_bbo = data_bbo.with_columns(
        ((pl.col('ask') - pl.col('bid')) / 2).alias("spread"),
        (pl.col('bid') * pl.col('bid_volume')).alias("bid_price_volume"),
        (pl.col('ask') * pl.col('ask_volume')).alias("ask_price_volume"),
        (pl.col('bid_volume') + pl.col('ask_volume')).alias("total_volume"),
    )

    data_bbo = data_bbo.with_columns(
        (pl.col('mid_price').log()).alias('log_mid_price'),  
        (pl.col('simple_return').log()).alias('log_return')
    )

    data_bbo = data_bbo.with_columns(
        pl.col('timestamp').cast(pl.Datetime).alias('timestamp')  # Ensure the 'Time' column is cast to Datetime
    )
    data_bbo = data_bbo.with_columns([
        # pl.col("timestamp").dt.year().alias("Year"),
        pl.col("timestamp").dt.month().alias("Month"),
        pl.col("timestamp").dt.day().alias("Day"),
        pl.col("timestamp").dt.hour().alias("Hour"),
        pl.col("timestamp").dt.minute().alias("Minute"),
        # pl.col("timestamp").dt.second().alias("Second"),
    ])
    
    return data_bbo

def main():
    currentPath = os.getcwd()
    data_bbo = final_data_bbo(f"{currentPath}/data/cleaned_bbo_data.parquet")
    data_bbo.write_parquet(f"{currentPath}/data/final_bbo_data.parquet")
    
if __name__ == "__main__":
    main()