import sys
sys.path.append("/project/")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, PercentFormatter
import seaborn as sns

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
    #"ant_uni_1000",
    "hopper_uni_250",
    #"hopper_uni_1000",
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

INIT_ALGO_LIST = [
    "mcpg_me_",
    #"mcpg_me_"
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
    "mcpg_me_fixed",
]

ALGO_LIST = [
    "mcpg_me_",
    "mcpg_me_orth_0",
    "mcpg_me_orth_05",
    "mcpg_me_unif_0",
    "mcpg_me_unif_05",
    "mcpg_me_unif_1",
]

ALGO_DICT = {
    "mcpg_me_0": "MCPG-ME 0% GA",
    "mcpg_me_0.25": "MCPG-ME 25% GA",
    "mcpg_me_0.5": "MCPG-ME 50% GA",
    "mcpg_me_0.75": "MCPG-ME 75% GA",
    "mcpg_me_1": "MCPG-ME 100% GA",
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
    "mcpg_me_fixed": "MCPG-ME fixed",
    "mcpg_me_orth_0": "MCPG-ME orth 0",
    "mcpg_me_orth_05": "MCPG-ME orth 0.5",
    "mcpg_me_unif_0": "MCPG-ME unif 0",
    "mcpg_me_unif_05": "MCPG-ME unif 0.5",
    "mcpg_me_unif_1": "MCPG-ME unif 1",
}

EMITTER_LIST = {
    "mcpg_me_": ["ga_offspring_added", "qpg_offspring_added"],
    "mcpg_me_orth_0": ["ga_offspring_added", "qpg_offspring_added"],
    "mcpg_me_orth_05": ["ga_offspring_added", "qpg_offspring_added"],
    "mcpg_me_unif_0": ["ga_offspring_added", "qpg_offspring_added"],
    "mcpg_me_unif_05": ["ga_offspring_added", "qpg_offspring_added"],
    "mcpg_me_unif_1": ["ga_offspring_added", "qpg_offspring_added"],
    "dcg_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "pga_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "qd_pg": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "me": ["ga_offspring_added"],
    "me_es": ["es_offspring_added"],
}
EMITTER_DICT = {
    "ga_offspring_added": "GA",
    "qpg_offspring_added": "PG",
    #"dpg_offspring_added": "DPG",
    #"es_offspring_added": "ES",
    #"ai_offspring_added": "AI",
    #"qpg_ai_offspring_added": "PG + AI",
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
    nrows = len(EMITTER_LIST[ALGO_LIST[0]])
    fig, axes = plt.subplots(nrows=nrows, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 7))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the column
        axes[0, col].set_title(ENV_DICT[env])

        # Set the x label and formatter for the column
        axes[nrows-1, col].set_xlabel(XLABEL)
        axes[nrows-1, col].xaxis.set_major_formatter(formatter)

        # Get df for the current env
        df_plot = df[(df["env"] == env)]  

        # Plot
        for i, emitter in enumerate(EMITTER_LIST[ALGO_LIST[0]]):
            ax = sns.lineplot(
                df_plot,
                x="num_evaluations",
                y=emitter,
                hue="algo_",
                hue_order=ALGO_LIST,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[i, col],
            )

            if col == 0:
                axes[i, col].set_ylabel("Elites for {}".format(EMITTER_DICT[emitter]))
            else:
                axes[i, col].set_ylabel(None)

            # Customize axis
            customize_axis(axes[i, col])

    # Legend
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.04), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("fixed_nans/output/plot_main_emitters.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("fixed_nans/output/")
        
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Sum PG and AI emitters
    #df["qpg_ai_offspring_added"] = df["qpg_offspring_added"] + df["ai_offspring_added"]

    # Get cumulative sum of elites
    for emitter in EMITTER_DICT:
        df[emitter] = df.groupby(['env', 'algo', 'run'])[emitter].cumsum()

    # Filter
    df = df[df["algo"].isin(INIT_ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    df['algo_'] = df.apply(filter, axis=1)
    # Plot
    plot(df)
    