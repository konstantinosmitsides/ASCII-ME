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
import matplotlib.lines as mlines


from utils import get_df


# Define env and algo names
ENV_LIST = [
    # "ant_omni_250",
    # "anttrap_omni_250",
    #"humanoid_omni",
    "hopper_uni_250",
    "walker2d_uni_250",
    #"walker2d_uni_1000",
    #"halfcheetah_uni",
    "ant_uni_250",
    "anttrap_omni_250",
    "ant_omni_250",

    #"hopper_uni_1000",
    #"ant_uni_1000",
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

BATCH_LIST = [
    1024,
    4096,
    16384,
    32768,
]

ALGO_LIST = [
    'mcpg_me_ga_0_greedy_0',
    'mcpg_me_ga_25_greedy_0',
    'mcpg_me',
    # "mcpg_me_ga_05_greedy_1",
    'mcpg_me_ga_75_greedy_0',
    'mcpg_me_ga_1_greedy_0',
    # 'mcpg_me_1',
    #"pga_me",
]



ALGO_DICT = {
    "dcg_me": "DCRL",
    "dcg_me_": "DCRL",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    #"mcpg_me": "50% Iso+LineDD\n(ASCII-ME)",
    "mcpg_me": "ASCII-ME",
    #"mcpg_me": "Buffer\nSampling",
    "mcpg_me_ga_0_greedy_0": "0% Iso+LineDD",
    "mcpg_me_ga_25_greedy_0": "25% Iso+LineDD",
    "mcpg_me_ga_75_greedy_0": "75% Iso+LineDD",
    "mcpg_me_ga_1_greedy_0": "100% Iso+LineDD",
    "mcpg_me_ga_05_greedy_1": "ASCII-ME (Archive)",
    "memes": "MEMES",
    "ppga" : "PPGA",
    "mcpg_only": "MCPG-Only",
    "mcpg_only_05": "MCPG-Only (0.5)",
    "mcpg_only_1": "MCPG-Only (1)",
    "mcpg_me_05": "MCPG-ME (0.5)",
    "mcpg_me_1": "MCPG-ME (1)",
}

XLABEL = "Evaluations"

def filter(df_row):
    if df_row["algo"] == "pga_me":
        if df_row["batch_size"] != 1024:
            return 
    if df_row["algo"] == "ppga":
        if df_row["batch_size"] != 6000:
            return 
    if df_row["algo"] == "memes":
        if df_row["batch_size"] != 8192:
            return 
    if df_row["algo"] == "mcpg_me":
        if df_row["batch_size"] != 4096:
            return 
    
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 0:
            return "mcpg_me_ga_0_greedy_0"
        
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.25 and df_row["greedy"] == 0:
            return "mcpg_me_ga_25_greedy_0"

    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.75 and df_row["greedy"] == 0:
            return "mcpg_me_ga_75_greedy_0"
        
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 1 and df_row["greedy"] == 0:
            return "mcpg_me_ga_1_greedy_0"


    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 0.5:
            return "mcpg_me_ga_0_greedy_05"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 1:
            return "mcpg_me_ga_0_greedy_1"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.5 and df_row["greedy"] == 0.5:
            return "mcpg_me_ga_05_greedy_05"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.5 and df_row["greedy"] == 1:
            return "mcpg_me_ga_05_greedy_1"
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


def plot(summary_df):

    fig, axes = plt.subplots(nrows=1, ncols=1, sharex='col', figsize=(10, 5 * 0.7))
    color_palette = sns.color_palette("viridis", len(ALGO_LIST))

    #y_labels = ["QD Score after 1M evaluations"]

    for col in range(1):
        for row in range(1):
            ax = axes
            
            # Set title for each subplot
            # if row == 0:
            #     ax.set_title(f"{y_labels[col]}")
            
            # Formatter for the x-axis
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

            # Get df for the current env
            df_plot = summary_df
            
            if col == 0:
                sns.barplot(
                    data=df_plot,
                    x="env",
                    y="qd_score",
                    hue="algo",
                    hue_order=ALGO_LIST,
                    estimator=np.median,
                    errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                    ax=ax,
                    palette=color_palette,
                    legend=False,
                    dodge=True,
                    order=ENV_LIST,  # Changed from ALGO_LIST to ENV_LIST
                )
                ax.set_ylabel(f"QD Score")

            ax.set_xticklabels([ENV_DICT.get(env, env) for env in ENV_LIST], ha="center")
            ax.set_xlabel(None)
                
            # Customize the axis aesthetics
            customize_axis(ax)
            ax.set_axisbelow(True)
        

        # Update legend placement
        colors = sns.color_palette(palette=color_palette, n_colors=len(ALGO_LIST))
        patches = [mlines.Line2D([], [], color=colors[i], label=ALGO_DICT.get(algo, algo), linewidth=2.2, linestyle='-') for i, algo in enumerate(ALGO_LIST)]
        # Place legend to the right of the figure
        fig.legend(handles=patches, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)    
    
        fig.tight_layout()

    # Adjust the right margin to prevent legend cutoff
    #plt.subplots_adjust(right=0.78)

    # Save plot
    fig.savefig("fig4/output/GA_ablation_test.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("fig4/output/")
    #results_dir = Path("ablation/output/")
    #print(results_dir)
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_005_000]
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()

    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'qd_score']]
    #summary_df = summary_df[summary_df["batch_size"].isin(BATCH_LIST)]
    
    #df['env_'] = df.apply(filter_gpu_variants, axis=1)

    # Plot
    #plot(df)
    
    #plot_(df)
    
    plot(summary_df)

    # Plot
    #plot(df)
    
    
    
