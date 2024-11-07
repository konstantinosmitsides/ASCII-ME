import sys
sys.path.append("/project/")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.ticker as ticker


from utils import get_df


# Define env and algo names
ENV_LIST = [
    "A100",
    "L40S",
    #"ant_omni_250",
    #"anttrap_omni_250",
    #"humanoid_omni",
    #"walker2d_uni_250",
    #"walker2d_uni_1000",
    #"halfcheetah_uni",
    #"ant_uni_250",
    #"ant_uni_1000",
    #"hopper_uni_250",
    #"hopper_uni_1000",
    #"humanoid_uni",
]
ENV_DICT = {
    "A100": "GPU: A100",
    "L40S": "GPU: L40S",
    "ant_omni_250": " Ant Omni ",
    "anttrap_omni_250": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni_250": "Walker Uni",
    "walker2d_uni_1000": "Walker Uni",
    "halfcheetah_uni": "HalfCheetah Uni",
    "ant_uni_250": "   Ant Uni   ",
    "ant_uni_1000": "   Ant Uni   ",
    "humanoid_uni": "Humanoid Uni",
    "hopper_uni_250": "Hopper Uni",
    "hopper_uni_1000": "Hopper Uni",
}

BATCH_LIST = [
    1024,
    4096,
    16384,
    32768,
    #65536
    
]

ALGO_LIST = [
    #"dcg_me",
    #"dcg_me_gecco",
    #"pga_me",
    #"qd_pg",
    #"me",
    #"me_es",
    "mcpg_me",
    #"memes",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "mcpg_me": "MCPG-ME",
    "memes": "MEMES",
}

XLABEL = "Evaluations"


def filter_gpu_variants(df_row):
    if df_row["algo"] == "mcpg_me":
        if df_row["gpu"] == "A100":
            return 'A100'
        
        else:
            return 'L40S'
    else:
        return df_row["algo"]


def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax

def plot__(summary_df):
    # Determine the global maximum time across all data for uniform y-axis scaling
    max_time = summary_df['time'].max()

    fig, axes = plt.subplots(nrows=1, ncols=len(ENV_LIST), sharex=True, figsize=(15, 6))

    # Define a suitable color palette
    color_palette = sns.color_palette("viridis", len(BATCH_LIST))

    for idx, env in enumerate(ENV_LIST):
        ax = axes[idx]  # Now axes is a 1D array, as there's only one row
        
        # Get df for the current env
        df_plot = summary_df[summary_df["env_"] == env]
        
        sns.barplot(
            data=df_plot,
            x="algo",
            y="time",
            hue="batch_size",
            ax=ax,
            palette=color_palette,
            dodge=True
        )
        
        # Title and labels
        ax.set_title(f"Runtime (s) vs. Batch Size")
        ax.set_ylabel(None)
        ax.set_xticklabels([])
        ax.set_xlabel(ENV_DICT[env])
        
        # Set the same y-axis limits based on the max_time computed
        ax.set_ylim(0, max_time * 1.1)  # Adding 10% for better visualization at the top
        
        # Customize the axis aesthetics
        customize_axis(ax)

    # Adjust legend and layout
    colors = sns.color_palette(palette=color_palette, n_colors=len(BATCH_LIST))
    #patches = [mlines.Line2D([], [], color=colors[i], label=str(batch_size), linewidth=2.2, linestyle='-') for i, batch_size in enumerate(BATCH_LIST)]
    #fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncols=3, frameon=False)
    fig.align_ylabels(axes[:])  # Align y-labels
    fig.tight_layout()
    fig.savefig("A100/output/plot_main_.png", bbox_inches="tight")
    plt.close()




if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("A100/output/")
    #print(results_dir)
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 5_000_000]
    df['env_'] = df.apply(filter_gpu_variants, axis=1)

    
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()

    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score', 'batch_size', 'env_']]
    
    df['env_'] = df.apply(filter_gpu_variants, axis=1)

    # Plot
    #plot(df)
    
    #plot_(df)
    
    plot__(summary_df)