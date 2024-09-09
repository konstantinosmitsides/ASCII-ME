import sys
sys.path.append("/project/")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns
import matplotlib.patches as mpatches


from utils import get_df


# Define env and algo names
ENV_LIST = [
    #"ant_omni_250",
    #"anttrap_omni_250",
    #"humanoid_omni",
    #"walker2d_uni_250",
    "walker2d_uni_1000",
    #"halfcheetah_uni",
    #"ant_uni_250",
    "ant_uni_1000",
    #"hopper_uni_250",
    "hopper_uni_1000",
    #"humanoid_uni",
]
ENV_DICT = {
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
ALGO_LIST = [
    "mcpg_me",
    #"mcpg_me_no_normalizer",
    #"mcpg_me_no_baseline",
    #"mcpg_me_no_ppo_loss",
    "dcg_me",
    #"dcg_me_gecco",
    "pga_me",
    #"qd_pg",
    #"me_es",
    "memes",
    "me",
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
    "mcpg_me_no_normalizer": "MCPG-ME no normalizer",
    "mcpg_me_no_baseline": "MCPG-ME no baseline",
    "mcpg_me_no_ppo_loss": "MCPG-ME no PPO loss",
}

YLABEL = "Runtime (s)"


# Customize axis appearance
def customize_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(which="major", axis="y", linestyle='--', linewidth='0.5', color='0.9')
    return ax


def plot(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=len(ENV_LIST), squeeze=False, figsize=(25, 10))  # Adjust the size as needed
    
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    
    for col, env in enumerate(ENV_LIST):
        ax = axes[0, col]
        
        # Set title for each subplot
        ax.set_title(ENV_DICT[env])
        ax.xaxis.set_major_formatter(formatter)

        
        # Get df for the current env
        df_plot = df[df["env"] == env]
        ax.yaxis.set_major_formatter(formatter)
        
        # Create boxplot for 'time' of each algorithm
        #sns.boxplot(
        #    data=df_plot,
        #    x="algo",
        #    y="time",
        #    hue="algo",
        #    hue_order=ALGO_LIST,
        #    order=ALGO_LIST,
        #    legend=False,
        #    ax=ax,
        #    #palette="vlag"
        #)

        
        sns.barplot(
            data=df_plot,
            x="algo",
            y="time",
            estimator=np.median,  
            errorbar=None,
            ax=ax,
            hue="algo",
            hue_order=ALGO_LIST,  
            order=ALGO_LIST,  
            legend=False,  
        )
        
        

        # Set the x-axis labels (Algorithm names) with rotation for better visibility
        #ax.set_xticklabels([ALGO_DICT[algo] for algo in ALGO_LIST], rotation=45, ha="right")
        
        # Label the y-axis with 'Time (s)' for the first subplot only
        ax.set_ylim(0.0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_xticks([])
        # Customize the axis aesthetics
        customize_axis(ax)
    colors = sns.color_palette(n_colors=len(ALGO_LIST))  # Get a color palette with 3 distinct colors
    patches = [mpatches.Patch(color=colors[i], label=ALGO_DICT[algo]) for i, algo in enumerate(ALGO_LIST)]
    #fig.legend(ax_.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncols=len(ALGO_LIST), frameon=False)
    # Adjust layout and set a common y-axis label
    fig.supylabel(YLABEL, y=0.6)  # Adjust vertical position if necessary
    fig.align_ylabels(axes[:, 0])  
    fig.tight_layout()

    # Save the plot
    fig.savefig("data_time_efficiency/output/time_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("data_time_efficiency/output/")
    #print(results_dir)
    EPISODE_LENGTH = 1000
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    
    
    # Get the median time for each (env, algo)

    #idx = df.groupby(["env", "algo", "run"]).apply(
    #    lambda x: min(x['iteration'].idxmax(), x.index.min() + 1999, x.index.max())
    #)
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score']]
    
    plot(summary_df)