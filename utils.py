import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA

def plot_two_risks(risks_df_in_sample, risks_df_out_sample):
    """
    Plot In-Sample vs Out-of-Sample Risk with customization.

    Parameters:
        risks_df_in_sample (pd.DataFrame): DataFrame containing in-sample risk data with a "Risk" column.
        risks_df_out_sample (pd.DataFrame): DataFrame containing out-of-sample risk data with a "Risk" column.
    """
    
    sns.set_theme(style="whitegrid", palette="muted", font="serif")

    # Plot the risks
    plt.figure(figsize=(14, 7))
    sns.lineplot(
        x=risks_df_out_sample.index,
        y=risks_df_out_sample["Risk"],
        label="Out-of-Sample Risk",
        marker="o",
        markersize=8,
        linewidth=2.5,
        color="#4C72B0"
    )
    sns.lineplot(
        x=risks_df_out_sample.index,
        y=risks_df_in_sample["Risk"],
        label="In-Sample Risk",
        marker="s",
        markersize=8,
        linewidth=2.5,
        color="#DD8452"
    )

    # Customize the plot
    plt.title("In-Sample vs Out-of-Sample Risk Over Time", fontsize=18, weight="bold")
    plt.xlabel("Time", fontsize=20, labelpad=10)
    plt.ylabel("Risk", fontsize=20, labelpad=10)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13, title="Risk Type", title_fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([0, 1e-6])  # Specify y-axis limits
    plt.xticks(ticks=[], labels=[])  # Remove x-axis tick labels
    plt.tight_layout()

    # Add a horizontal line for emphasis (optional)
    plt.axhline(y=5e-7, color="gray", linestyle="--", linewidth=1.5, alpha=0.8)

    # Show the plot
    return plt

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