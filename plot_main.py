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
    "ant_omni_250",
    "anttrap_omni_250",
    #"humanoid_omni",
    "walker2d_uni_250",
    #"walker2d_uni_1000",
    #"halfcheetah_uni",
    "ant_uni_250",
    "hopper_uni_250",
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
    #"mcpg_me",
    #"mcpg_me__"
    "mcpg_me_",
    #"mcpg_me_no_normalizer",
    #"mcpg_me_no_baseline",
    #"mcpg_me_no_ppo_loss",
    #"dcg_me",
    #"dcg_me_"
    #"dcg_me_gecco",
    #"pga_me",
    #"qd_pg",
    #"me_es",
    #"memes",
    #"me",
    #"me_2",
    "mcpg_me_fixed",
]

NEW_ALGO_LIST = [
    "mcpg_me_",
    "mcpg_me_orth_0",
    "mcpg_me_orth_05",
    "mcpg_me_unif_0",
    "mcpg_me_unif_05",
    "mcpg_me_unif_1",
]

ALGO_DICT = {
    "me_0.05_0.05": "ME (0.05, 0.05)",
    "me_0.5_0.05": "ME (0.5, 0.05)",
    "me_0.005_0.5": "ME (0.005, 0.5)",
    "me_0.005_1.0": "ME (0.005, 1.0)",
    "me_0.05_0.5": "ME (0.05, 0.5)",
    "me_0.5_1.0": "ME (0.5, 1.0)",
    "dcg_me": "DCRL-AI-only",
    "dcg_me_": "DCRL",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "mcpg_me": "MCPG-ME",
    "memes": "MEMES",
    "mcpg_me_no_normalizer": "Ablation 1",
    "mcpg_me_no_baseline": "Ablation 2",
    "mcpg_me_no_ppo_loss": "Ablation 3",
    "mcpg_me_": "MCPG-ME old",
    #"mcpg_me_": "MCPG-ME orthogonal",
    "mcpg_me_fixed": "MCPG-ME fixed",
    "mcpg_me_orth_0": "MCPG-ME orth 0",
    "mcpg_me_orth_05": "MCPG-ME orth 0.5",
    "mcpg_me_unif_0": "MCPG-ME unif 0",
    "mcpg_me_unif_05": "MCPG-ME unif 0.5",
    "mcpg_me_unif_1": "MCPG-ME unif 1",
}

XLABEL = "Evaluations"

def filter(df_row):
    if df_row["algo"] == "mcpg_me_fixed":
        if df_row["init"] == "orthogonal" and df_row["greedy"] == 0:
            return "mcpg_me_orth_0"
        
        if df_row["init"] == "orthogonal" and df_row["greedy"] == 0.5:
            return "mcpg_me_orth_05"
        
        if df_row["init"] == "uniform" and df_row["greedy"] == 0:
            return "mcpg_me_unif_0"
        
        if df_row["init"] == "uniform" and df_row["greedy"] == 0.5:
            return "mcpg_me_unif_05"
        
        if df_row["init"] == "uniform" and df_row["greedy"] == 1:
            return "mcpg_me_unif_1"
            
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


def plot(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 10))

    # Create formatter
    #x_ticks = np.arange(0, 1_000_001, 500_000)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the column
        axes[0, col].set_title(ENV_DICT[env])

        # Set the x label and formatter for the column
        axes[0, col].set_xlabel(XLABEL)
        #axes[2, col].set_xticks(x_ticks)
        axes[0, col].xaxis.set_major_formatter(formatter)

        # Get df for the current env
        df_plot = df[df["env"] == env]

        # QD score
        axes[0, col].yaxis.set_major_formatter(formatter)

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="qd_score",
            hue="algo_",
            hue_order=NEW_ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[0, col],
        )

        if col == 0:
            axes[0, col].set_ylabel("QD score")
        else:
            axes[0, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[0, col])

        # Coverage
        axes[1, col].set_ylim(0., 1.05)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="coverage",
            hue="algo_",
            hue_order=NEW_ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[1, col],
        )
        axes[1, col].xaxis.set_major_formatter(formatter)


        if col == 0:
            axes[1, col].set_ylabel("Coverage")
        else:
            axes[1, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[1, col])

        # Max fitness
        ax = sns.lineplot(  # store ax for legend
            df_plot,
            x="num_evaluations",
            y="max_fitness",
            hue="algo_",
            hue_order=NEW_ALGO_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[2, col],
        )

        if col == 0:
            axes[2, col].set_ylabel("Max Fitness")
        else:
            axes[2, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[2, col])

    # Legend
    #fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)
    colors = sns.color_palette(n_colors=len(NEW_ALGO_LIST))  # Get a color palette with 3 distinct colors
    #patches = [mpatches.Patch(color=colors[i], label=ALGO_DICT[algo]) for i, algo in enumerate(ALGO_LIST)]
    #fig.legend(ax_.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)
    #fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.07), ncols=len(ALGO_LIST), frameon=False)
    patches = [mlines.Line2D([], [], color=colors[i], label=ALGO_DICT[algo], linewidth=2.2, linestyle='-') for i, algo in enumerate(NEW_ALGO_LIST)]
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(NEW_ALGO_LIST), frameon=False)
    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("fixed_nans/output/plot_main.png", bbox_inches="tight")
    #fig.savefig("ablation/output/plot_main.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("fixed_nans/output/")
    #results_dir = Path("ablation/output/")
    #print(results_dir)
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    
    df['algo_'] = df.apply(filter, axis=1)


    # Plot
    plot(df)
    
    
    
