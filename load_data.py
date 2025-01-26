import polars as pl
import os
import tarfile
import gzip
from pathlib import Path
import numpy as np
import os


def get_all_tickers_from_bbo(base_path = 'data/sp100_2004-8/bbo'):
    """
    Retrieves all folder names (tickers) in the 'bbo' directory and stores them in a set.
    
    Args:
        base_path (str): The path to the 'bbo' directory containing ticker folders.
    
    Returns:
        set: A set containing all stock tickers (folder names) in the 'bbo' directory.
    """
    tickers = set()
    
    # Check if the base path exists and is a directory
    if os.path.exists(base_path) and os.path.isdir(base_path):
        for folder_name in os.listdir(base_path):
            full_path = os.path.join(base_path, folder_name)
            #  make sure only directories are added
            if os.path.isdir(full_path):
                tickers.add(folder_name)
    
    return list(tickers)

def read_data_bbo(folder_type,
                           stock_list,
                           tz_exchange="America/New_York",
                           only_regular_trading_hours=True,
                           hhmmss_open="09:30:00",
                           hhmmss_close="16:00:00",
                           merge_same_index=True,
                           base_path='data/sp100_2004-8'):
    """
    Read data from BBO files for a list of stocks.

    Args:
        folder_type (): Indicates if we are reading from BBO or Trade folders.
        stock_list (list): List of stock symbols to read data for.
        tz_exchange (str, optional): _description_. Defaults to "America/New_York".
        only_regular_trading_hours (bool, optional): Trading hour. Defaults to True.
        hhmmss_open (str, optional): Open Time of Trading. Defaults to "09:30:00".
        hhmmss_close (str, optional): Closing Time of Trading. Defaults to "16:00:00".
        merge_same_index (bool, optional): Defaults to True.
        base_path (str, optional): Path where data is stored. Defaults to 'data/sp100_2004-8'.

    Raises:
        ValueError: Errors in the input.

    Returns:
        _type_: Polars DataFrame containing the BBO data.
    """
    if not isinstance(stock_list, (list, set)):
        raise ValueError(f"Expected stock_list to be a list or set, but got {type(stock_list)}.")

    # to store dataframes
    stock_dfs = {}
    
    # path to specific folder type
    folder_path = Path(base_path) / folder_type
    
    stock_set = set(stock_list)

    
    for stock_folder in folder_path.iterdir():
        if stock_folder.is_dir():
            stock_symbol = stock_folder.name
            
            if stock_symbol not in stock_set:
                continue
            
            # Find .tar file in the stock folder
            tar_files = list(stock_folder.glob("*.tar"))
            if not tar_files:
                print(f"No tar file found for {stock_symbol}")
                continue
                
            tar_file = tar_files[0]
            
            # List to store individual dataframes for concatenation
            dfs_to_concat = []
            
            with tarfile.open(tar_file, 'r') as tar:
                # all .csv.gz files starting from July 2008
                months = ['2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12']
                csv_gz_files = [
                    f for f in tar.getmembers()
                    if any(f.name.startswith(month) for month in months) and f.name.endswith('.csv.gz')
                ]
                
                for csv_gz_file in csv_gz_files:
                    f = tar.extractfile(csv_gz_file)
                    if f is not None:
                        # Decompress the content
                        with gzip.open(f, 'rb') as gz_file:
                            try:
                                df = pl.read_csv(
                                    gz_file,
                                    null_values=["()"],
                                    #schema=schema,
                                    low_memory=True
                                )
                                # column for the stock symbol
                                df = df.with_columns(pl.lit(stock_symbol).alias("Stock"))
                                dfs_to_concat.append(df)
                            except Exception as e:
                                print(f"Error reading {csv_gz_file.name}: {str(e)}")
            
            # Concatenate all dataframes for this stock
            if dfs_to_concat:
                try:
                    stock_dfs[stock_symbol] = pl.concat(
                        dfs_to_concat,
                        how="vertical_relaxed"
                    )
                except Exception as e:
                    print(f"Error concatenating dataframes for {stock_symbol}: {str(e)}")
    
    # Concatenate all DataFrames into one
    all_dataframes = []
    for df in stock_dfs.values():
        all_dataframes.append(df)
    
    DF = pl.concat(all_dataframes, how="vertical_relaxed") if all_dataframes else pl.DataFrame()
    
    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    DF.drop("xltime")

    # apply common sense filter
    DF = DF.filter(pl.col("ask-price")>0).filter(pl.col("bid-price")>0).filter(pl.col("ask-price")>pl.col("bid-price"))

    if merge_same_index:
        DF = DF.group_by('index', maintain_order=True).last()   # last quote of the same timestamp
    
    if only_regular_trading_hours:
        hh_open,mm_open,ss_open = [float(x) for x in hhmmss_open.split(":")]
        hh_close,mm_close,ss_close = [float(x) for x in hhmmss_close.split(":")]

        seconds_open=hh_open*3600+mm_open*60+ss_open
        seconds_close=hh_close*3600+mm_close*60+ss_close

        DF = DF.filter(pl.col('index').dt.hour().cast(float)*3600+pl.col('index').dt.minute().cast(float)*60+pl.col('index').dt.second()>=seconds_open,
                       pl.col('index').dt.hour().cast(float)*3600+pl.col('index').dt.minute().cast(float)*60+pl.col('index').dt.second()<=seconds_close)
    
    return DF
    

def read_data_trade(folder_type,
                           stock_list,
                           tz_exchange="America/New_York",
                           only_non_special_trades=True,
                           only_regular_trading_hours=True,
                           merge_sub_trades=True,
                           base_path='data/sp100_2004-8'):
    """
    Read data from Trade files for a list of stocks.

    Args:
        folder_type (str): Indicates if we are reading from BBO or Trade folders.
        stock_list (list): List of stock symbols to read data for.
        tz_exchange (str, optional): Exchange Rate taken into consideration. Defaults to "America/New_York".
        only_non_special_trades (bool, optional): Defaults to True.
        only_regular_trading_hours (bool, optional): Consider only regular trading hours. Defaults to True.
        merge_sub_trades (bool, optional): Merge Trades for Different stocks. Defaults to True.
        base_path (str, optional): Path where data is stored. Defaults to 'data/sp100_2004-8'.

    Raises:
        ValueError: Errors in the input.

    Returns:
        _type_: Polars DataFrame containing the Trade data.
    """
    
    
    # Validate input
    if not isinstance(stock_list, (list, set)):
        raise ValueError(f"Expected stock_list to be a list or set, but got {type(stock_list)}.")

    # to store dataframes
    stock_dfs = {}
    
    # path to specific folder type
    folder_path = Path(base_path) / folder_type
    
    stock_set = set(stock_list)

    
    for stock_folder in folder_path.iterdir():
        if stock_folder.is_dir():
            stock_symbol = stock_folder.name
            
            # Skip if stock is not in the specified list
            if stock_symbol not in stock_set:
                continue
            
            # Find .tar file in the stock folder
            tar_files = list(stock_folder.glob("*.tar"))
            if not tar_files:
                print(f"No tar file found for {stock_symbol}")
                continue
                
            tar_file = tar_files[0]
            
            # List to store individual dataframes for concatenation
            dfs_to_concat = []
            
            # Open and process tar file
            with tarfile.open(tar_file, 'r') as tar:
                # .csv.gz files starting from Jluy 2008
                months = ['2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12']
                csv_gz_files = [
                    f for f in tar.getmembers()
                    if any(f.name.startswith(month) for month in months) and f.name.endswith('.csv.gz')
                ]
                
                for csv_gz_file in csv_gz_files:
                    f = tar.extractfile(csv_gz_file)
                    if f is not None:
                        # Decompress the content
                        with gzip.open(f, 'rb') as gz_file:
                            try:
                                df = pl.read_csv(
                                    gz_file,
                                    null_values=["()"],
                                    schema_overrides={"trade-volume": pl.Float64},
                                    #schema=schema,
                                    low_memory=True
                                )
                                # Add a column for the stock symbol
                                df = df.with_columns(pl.lit(stock_symbol).alias("Stock"))
                                dfs_to_concat.append(df)
                            except Exception as e:
                                print(f"Error reading {csv_gz_file.name}: {str(e)}")
            
            # Concatenate all dataframes for this stock
            if dfs_to_concat:
                try:
                    stock_dfs[stock_symbol] = pl.concat(
                        dfs_to_concat,
                        how="vertical_relaxed"
                    )
                except Exception as e:
                    print(f"Error concatenating dataframes for {stock_symbol}: {str(e)}")
    
    # Concatenate all DataFrames into one
    all_dataframes = []
    for df in stock_dfs.values():
        all_dataframes.append(df)
    
    DF = pl.concat(all_dataframes, how="vertical_relaxed") if all_dataframes else pl.DataFrame()
    
    excel_base_date = pl.datetime(1899, 12, 30)  # Excel starts counting from 1900-01-01, but Polars needs 1899-12-30
    DF = DF.with_columns(
        (pl.col("xltime") * pl.duration(days=1) + excel_base_date).alias("index")
    )
    DF = DF.with_columns(pl.col("index").dt.convert_time_zone(tz_exchange))
    DF.drop(["xltime","trade-rawflag","trade-stringflag"])

    if only_non_special_trades:
        DF=DF.filter(pl.col("trade-stringflag")=="uncategorized")

    if merge_sub_trades:   # average volume-weighted trade price here
        DF=DF.group_by('index',maintain_order=True).agg([(pl.col('trade-price')*pl.col('trade-volume')).sum()/(pl.col('trade-volume').sum()).alias('trade-price'),pl.sum('trade-volume')])        
    
    return DF


def main():
    currentPath = os.getcwd()
    tickers = get_all_tickers_from_bbo(f"{currentPath}/data/sp100_2004-8/bbo")
    subset_tickers = tickers[10:22]
    len(subset_tickers)
    data_bbo = read_data_bbo('bbo', subset_tickers)
    # data_trade = read_data_trade('trade', tickers)
    data_bbo.write_parquet(f"{currentPath}/data/raw_small_bbo_data.parquet")
    # data_trade.write_parquet(f"{currentPath}/data/raw_trade_data_1.parquet")
    
if __name__ == "__main__":
    main()