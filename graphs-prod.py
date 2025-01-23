#!/usr/bin/env python3

"""

A  script for generating exploratory plots from the cleaned Parquet data.
It uses Polars for data I/O and Matplotlib/Seaborn for plotting. 


"""

import os
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def create_plots_directory(base_path: str = "plots") -> str:
    """
    Ensures the existence of a directory for saving plot images,
    returning its absolute path.
    """
    plots_dir = os.path.abspath(base_path)
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def load_data(script_dir: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Loads the cleaned BBO and trade data from Parquet files
    in the data folder within the project structure.
    """
    bbo_path = os.path.join(script_dir, "data", "cleaned_bbo_data.parquet")
    trade_path = os.path.join(script_dir, "data", "cleaned_trade_data.parquet")
    bbo_df = pl.read_parquet(bbo_path)
    trade_df = pl.read_parquet(trade_path)
    return bbo_df, trade_df

def plot_time_series(df: pl.DataFrame, time_col: str, value_col: str, title: str, output_path: str) -> None:
    pdf = df.to_pandas()
    pdf.sort_values(by=time_col, inplace=True)

    plt.figure(figsize=(10, 5))
    plt.plot(pdf[time_col], pdf[value_col], label=value_col, color="blue")
    plt.title(title)
    plt.xlabel(time_col)
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_histogram(df: pl.DataFrame, col: str, title: str, output_path: str, bins: int = 50) -> None:
    pdf = df.to_pandas()
    plt.figure(figsize=(6, 4))
    sns.histplot(pdf[col], bins=bins, kde=True, color="green")
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_boxplot(df: pl.DataFrame, col: str, title: str, output_path: str) -> None:
    pdf = df.to_pandas()
    plt.figure(figsize=(4, 6))
    sns.boxplot(x=pdf[col], orient="v", color="lightblue")
    plt.title(title)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scatter(df: pl.DataFrame, x_col: str, y_col: str, title: str, output_path: str) -> None:
    pdf = df.to_pandas()
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=pdf, x=x_col, y=y_col, color="purple", alpha=0.7)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correlation_heatmap(df: pl.DataFrame, columns: list[str], title: str, output_path: str) -> None:
    pdf = df.select(columns).to_pandas()
    corr_matrix = pdf.corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pairplot(df: pl.DataFrame, columns: list[str], title: str, output_path: str) -> None:
    """
    Generates a Seaborn pairplot for the specified columns.
    Allows for multi-dimensional visualization of relationships.
    """
    pdf = df.select(columns).to_pandas()
    g = sns.pairplot(pdf, corner=True, diag_kind="kde")
    g.fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """
    Main function to orchestrate data loading, progress logging, 
    and creation of multiple plots saved to a dedicated directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = create_plots_directory(os.path.join(script_dir, "plots"))

    bbo_df, trade_df = load_data(script_dir)

    # Define a sequence of plot tasks
    tasks = [
        {
            "func": plot_time_series,
            "kwargs": {
                "df": bbo_df,
                "time_col": "timestamp",
                "value_col": "bid",
                "title": "BBO: Bid Over Time",
                "output_path": os.path.join(plots_dir, "bbo_bid_over_time.png")
            },
            "desc": "Plot BBO Bid Over Time"
        },
        {
            "func": plot_time_series,
            "kwargs": {
                "df": bbo_df,
                "time_col": "timestamp",
                "value_col": "ask",
                "title": "BBO: Ask Over Time",
                "output_path": os.path.join(plots_dir, "bbo_ask_over_time.png")
            },
            "desc": "Plot BBO Ask Over Time"
        },
        {
            "func": plot_histogram,
            "kwargs": {
                "df": trade_df,
                "col": "price",
                "title": "Trade Price Distribution",
                "output_path": os.path.join(plots_dir, "trade_price_histogram.png"),
                "bins": 50
            },
            "desc": "Plot Trade Price Histogram"
        },
        {
            "func": plot_boxplot,
            "kwargs": {
                "df": trade_df,
                "col": "volume",
                "title": "Trade Volume Boxplot",
                "output_path": os.path.join(plots_dir, "trade_volume_boxplot.png")
            },
            "desc": "Plot Trade Volume Boxplot"
        },
        {
            "func": plot_scatter,
            "kwargs": {
                "df": trade_df,
                "x_col": "price",
                "y_col": "volume",
                "title": "Price vs. Volume Scatter",
                "output_path": os.path.join(plots_dir, "trade_price_vs_volume_scatter.png")
            },
            "desc": "Plot Price vs. Volume Scatter"
        },
        {
            "func": plot_correlation_heatmap,
            "kwargs": {
                "df": trade_df,
                "columns": ["price", "volume"],
                "title": "Trade Data Correlation Heatmap",
                "output_path": os.path.join(plots_dir, "trade_corr_heatmap.png")
            },
            "desc": "Plot Trade Correlation Heatmap"
        },
        {
            "func": plot_pairplot,
            "kwargs": {
                "df": trade_df,
                "columns": ["price", "volume"],
                "title": "Trade Data Pairplot",
                "output_path": os.path.join(plots_dir, "trade_data_pairplot.png")
            },
            "desc": "Plot Trade Pairplot (price, volume)"
        }
    ]

    # Iterate over the tasks with a progress bar
    for task in tqdm(tasks, desc="Generating Plots", unit="plot"):
        task["func"](**task["kwargs"])

    print(f"\nAll plots have been generated and saved in: {plots_dir}")

if __name__ == "__main__":
    main()
