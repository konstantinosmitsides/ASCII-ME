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
    "hopper_uni",
    "walker2d_uni",
    "ant_uni",
    "anttrap_omni",
    "ant_omni",
]

ENV_DICT = {
    "ant_omni": " Ant Omni ",
    "anttrap_omni": "AntTrap Omni",
    "humanoid_omni": "Humanoid Omni",
    "walker2d_uni": "Walker Uni",
    "ant_uni": "   Ant Uni   ",
    "hopper_uni": "Hopper Uni",
}


ALGO_LIST = [
    "ascii_me",
    "dcrl_me",
    "pga_me",
    "me",
]


ALGO_DICT = {
    "ascii_me": "ASCII-ME",
    "dcrl_me": "DCRL-ME",
    "pga_me": "PGA-ME",
    "me": "ME",
}

EMITTER_LIST = {
    "ascii_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "dcrl_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "pga_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
}
EMITTER_DICT = {
    "ga_offspring_added": "Iso+LineDD",
    "qpg_ai_offspring_added": "PG + AI",
}

XLABEL = "Evaluations"

def filter(df_row):
    if df_row["algo"] == "pga_me":
        if df_row["batch_size"] != 1024:
            return 

    if df_row["algo"] == "me":
        if df_row["batch_size"] != 8192:
            return 
        
    if df_row["algo"] == "ppga":
        if df_row["batch_size"] != 6000:
            return 
        
    if df_row["algo"] == "memes":
        if df_row["batch_size"] != 8192:
            return 
        
    if df_row["algo"] == "dcrl_me":
        if df_row["batch_size"] != 2048:
            return 
        
    if df_row["algo"] == "ascii_me":
        if df_row["batch_size"] != 4096:
            return 
        

        
    if df_row["algo"] == "ascii_me":
        if df_row["proportion_mutation_ga"] == 0:
            return "ascii_100"
    if df_row["algo"] == "ascii_me":
        if df_row["proportion_mutation_ga"] == 0.25:
            return "ascii_75"
        
    if df_row["algo"] == "ascii_me":
        if df_row["proportion_mutation_ga"] == 0.75:
            return "ascii_25"

    if df_row["algo"] == "ascii_me":
        if df_row["proportion_mutation_ga"] == 1:
            return "ascii_0"

        

    return df_row["algo"]

def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    # Apply scientific notation to y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)
    return ax


def plot(df):
    # Create subplots
    nrows = len(EMITTER_LIST[ALGO_LIST[1]])
    fig, axes = plt.subplots(nrows=nrows, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25 * 0.8, 7 * 0.7))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    # Flag to handle legend retrieval only once
    retrieved_legend = False
    handles, labels = None, None

    for col, env in enumerate(ENV_LIST):
        print(env)
        axes[0, col].set_title(ENV_DICT[env])
        axes[nrows-1, col].set_xlabel(XLABEL)
        axes[nrows-1, col].xaxis.set_major_formatter(formatter)

        # Filter df for the current env
        df_plot = df[(df["env"] == env)]

        for i, emitter in enumerate(EMITTER_LIST[ALGO_LIST[1]]):
            # Only let the very first subplot create a legend
            if not retrieved_legend and i == 0 and col == 0:
                ax = sns.lineplot(
                    data=df_plot,
                    x="num_evaluations",
                    y=emitter,
                    hue="algo",
                    hue_order=ALGO_LIST,
                    estimator=np.median,
                    errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                    ax=axes[i, col],
                )
                handles, labels = axes[i, col].get_legend_handles_labels()
                # Remove the local legend after retrieving handles and labels
                axes[i, col].get_legend().remove()
                retrieved_legend = True
            else:
                # For all other plots, disable the local legend
                ax = sns.lineplot(
                    data=df_plot,
                    x="num_evaluations",
                    y=emitter,
                    hue="algo",
                    hue_order=ALGO_LIST,
                    estimator=np.median,
                    errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                    legend=False,
                    ax=axes[i, col],
                )

            if col == 0:
                label = EMITTER_DICT.get(emitter, None)
                if label is not None:
                    axes[i, col].set_ylabel("Elites for \n{}".format(label))
            else:
                axes[i, col].set_ylabel(None)

            # Customize axis
            customize_axis(axes[i, col])

    # Map the original labels (which are algo names) through ALGO_DICT
    mapped_labels = [ALGO_DICT.get(l, l) for l in labels]

    # Create a single global legend at the bottom
    fig.legend(handles, mapped_labels, loc="lower center", bbox_to_anchor=(0.5, -0.04),
               ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("output/fig4.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("output/")
        
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    df['algo'] = df.apply(filter, axis=1)


    # Sum PG and AI emitters
    df['ai_offspring_added'] = df['ai_offspring_added'].fillna(0)
    df["qpg_ai_offspring_added"] = df["qpg_offspring_added"] + df["ai_offspring_added"]

    # Get cumulative sum of elites
    for emitter in EMITTER_DICT:
        df[emitter] = df.groupby(['env', 'algo', 'run'])[emitter].cumsum()

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]

    # Plot
    plot(df)
    