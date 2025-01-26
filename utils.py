import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_moving_average_returns(parquet_file, T_in, dT, in_sample_period, out_sample_period):
    # Load and preprocess data
    data = pd.read_parquet(parquet_file)
    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values(by=["time", "Stock"])

    #  pivot table
    pivot_table = data.pivot(index="time", columns="Stock", values="vwap_mid_price")
    pivot_table = pivot_table.sort_index(axis=1)
    pivot_table = pivot_table.fillna(method='bfill', axis=0).fillna(method='ffill', axis=0)

    # log-returns
    returns = pivot_table.pct_change()
    log_rets = np.log(1 + returns).fillna(0)

    #  rolling means
    means_rolling_data = []
    ts = np.arange(0, len(log_rets), dT)
    valid_indices = [t for t in ts if t + T_in <= len(log_rets)]  # Ensure indices are valid
    for t0 in valid_indices:
        t1 = t0 + T_in
        log_rets_cut = log_rets.iloc[t0:t1]
        means_rolling_data.append(log_rets_cut.mean())

    # DataFrame for moving averages
    means_rolling_data_df = pd.DataFrame(
        means_rolling_data,
        index=log_rets.index[valid_indices],
        columns=pivot_table.columns
    )

    # Plot the moving averages
    plt.figure(figsize=(16, 8))
    for Stock in means_rolling_data_df.columns:
        plt.plot(
            means_rolling_data_df.index,
            means_rolling_data_df[Stock],
            label=Stock,
            linewidth=1
        )

    # Customize the plot
    plt.title("Moving Average Returns of Stocks", fontsize=20, weight='bold')
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Moving Average Log-Returns", fontsize=16)
    plt.legend(fontsize=12, title="Tickers", title_fontsize=13, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gcf().autofmt_xdate()
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvspan(*in_sample_period, color='green', alpha=0.1, label='In-Sample Period')
    plt.axvspan(*out_sample_period, color='yellow', alpha=0.1, label='Out-of-Sample Period')
    plt.tight_layout()
    plt.ylim([-0.02, 0.02])
    plt.show()
    
def plot_two_risks_clipped(risks_df_in_sample, risks_df_out_sample):
    """
    Plot In-Sample vs Out-of-Sample Risk with improved visualization.

    Parameters:
        risks_df_in_sample (pd.DataFrame): DataFrame containing in-sample risk data with a "Risk" column.
        risks_df_out_sample (pd.DataFrame): DataFrame containing out-of-sample risk data with a "Risk" column.
    """
    sns.set_theme(style="whitegrid", palette="muted", font="serif")

    # Create a range for the x-axis
    x_range = range(len(risks_df_out_sample))

    # Plot the risks
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        x=x_range,
        y=risks_df_in_sample["Risk"],
        label="In-Sample Risk",
        linewidth=1.5,
        color="blue"
    )
    sns.lineplot(
        x=x_range,
        y=risks_df_out_sample["Risk"],
        label="Out-of-Sample Risk",
        linewidth=1.5,
        color="red"
    )

    # Customize the plot
    plt.title("In-Sample vs Out-of-Sample Variance Over Time", fontsize=16, weight="bold")
    plt.xlabel("Time", fontsize=14, labelpad=10)
    plt.ylabel("Risk", fontsize=14, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12, title="Risk Type", title_fontsize=12, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0, 5e-5])  # Specify y-axis limits
    plt.tight_layout()

    # Add a horizontal line for emphasis (optional)
    plt.axhline(y=5e-7, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    # Show the plot
    plt.show()

def plot_two_risks(risks_df_in_sample, risks_df_out_sample):
    """
    Plot In-Sample vs Out-of-Sample Risk with improved visualization.

    Parameters:
        risks_df_in_sample (pd.DataFrame): DataFrame containing in-sample risk data with a "Risk" column.
        risks_df_out_sample (pd.DataFrame): DataFrame containing out-of-sample risk data with a "Risk" column.
    """
    sns.set_theme(style="whitegrid", palette="muted", font="serif")

    # Create a range for the x-axis
    x_range = range(len(risks_df_out_sample))

    # Plot the risks
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        x=x_range,
        y=risks_df_in_sample["Risk"],
        label="In-Sample Risk",
        linewidth=1.5,
        color="blue"
    )
    sns.lineplot(
        x=x_range,
        y=risks_df_out_sample["Risk"],
        label="Out-of-Sample Risk",
        linewidth=1.5,
        color="red"
    )

    # Customize the plot
    plt.title("In-Sample vs Out-of-Sample Variance Over Time", fontsize=16, weight="bold")
    plt.xlabel("Time", fontsize=14, labelpad=10)
    plt.ylabel("Risk", fontsize=14, labelpad=10)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12, title="Risk Type", title_fontsize=12, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0, 5e-6])  # Specify y-axis limits
    plt.tight_layout()

    # Add a horizontal line for emphasis (optional)
    plt.axhline(y=5e-7, color="gray", linestyle="--", linewidth=1, alpha=0.8)

    # Show the plot
    plt.show()

def compute_weights_GVM(covariance_matrix):
    """
    Function to compute the Global Minimum Variance (GMV) portfolio weights.

    Args:
        Sigma (): Covariance matrix of the asset returns.

    Returns:
        _type_: _description_
    """
    covariance_matrix_inv = LA.inv(covariance_matrix)
    weights = covariance_matrix_inv.sum(axis=1) / covariance_matrix_inv.sum()
    return weights


def eigenvalue_clipping(lambdas, v, lambda_plus):
    N=len(lambdas)
    
    
    # _s stands for _structure below
    sum_lambdas_gt_lambda_plus = np.sum(lambdas[lambdas>lambda_plus])
    
    sel_bulk = lambdas<=lambda_plus                     
    N_bulk=np.sum(sel_bulk)
    sum_lambda_bulk=np.sum(lambdas[sel_bulk])        
    delta=sum_lambda_bulk/N_bulk                      
    
    lambdas_clean=lambdas
    lambdas_clean[lambdas_clean<=lambda_plus]=delta
    
    
    C_clean=np.zeros((N, N))
    v_m=np.matrix(v)
    
    for i in range(N-1):
        C_clean=C_clean+lambdas_clean[i] * np.dot(v_m[i,].T,v_m[i,]) 
        
    np.fill_diagonal(C_clean,1)
            
    return C_clean    
    
def P0(lambdas,q, sigma):
    lambda_plus = sigma**2*(1+np.sqrt(q))**2
    lambda_minus = sigma**2*(1-np.sqrt(q))**2
    vals = 1/(q*2*np.pi*sigma**2*lambdas)*np.sqrt((lambda_plus-lambdas)*(lambdas-lambda_minus))
    return vals