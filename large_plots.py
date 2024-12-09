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

ENV_LIST = [
    "ant_omni_250",
    "anttrap_omni_250",
    "walker2d_uni_250",
    "ant_uni_250",
    "hopper_uni_250",
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
    "mcpg_me_fixed",
]

NEW_ALGO_LIST = [
    "mcpg_me_no_clipping",
    "mcpg_me_normal",
]

ALGO_DICT = {
    "mcpg_me_no_clipping": "MCPG-ME no clip",
    "mcpg_me_normal": "MCPG-ME with clip",
}

XLABEL = "Evaluations"


def filter(df_row):
    if df_row["algo"] == "mcpg_me_fixed":
        
        if df_row["learning_rate"] == 0.3 or df_row["learning_rate"] == 0.0003:
            return "mcpg_me_no_clipping"
    return df_row["algo"]


def customize_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(which="major", axis="y", color="0.9")
    return ax


def plot(df):
    df["clip_param"] = df["clip_param"].astype(str)

    run_group_cols = ["env", "clip_param", "std", "learning_rate", "no_epochs", "run"]
    # # Instead of taking the last value, take the median of the performance metrics
    # df_final = (
    #     df.groupby(group_cols, as_index=False)
    #     .median(numeric_only=True)  # median of numeric columns only
    # )

    # Sort by num_evaluations and take the last entry for each run
    df_last = df.sort_values("num_evaluations").groupby(run_group_cols, as_index=False).last()

    # Now group by configuration (excluding 'seed') and take the median of the last values
    config_group_cols = ["env", "clip_param", "std", "learning_rate", "no_epochs"]
    df_final = df_last.groupby(config_group_cols, as_index=False).median(numeric_only=True)

    fig, axes = plt.subplots(nrows=3, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 10))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        axes[0, col].set_title(ENV_DICT[env])
        axes[0, col].xaxis.set_major_formatter(formatter)

        df_plot = df_final[df_final["env"] == env]

        # For the first subplot (0,0), allow legend. For others, disable.
        show_legend = (col == 0)

        # QD Score
        sns.scatterplot(
            data=df_plot,
            x="std",
            y="qd_score",
            hue="learning_rate",
            style="clip_param",
            markers={"0.2": "o", "0.0": "X"},
            palette="deep",
            ax=axes[0, col],
            s=80,
            alpha=0.6,
            legend="full" if show_legend else False,
        )

        if col == 0:
            axes[0, col].set_ylabel("QD score")
        else:
            axes[0, col].set_ylabel(None)
        customize_axis(axes[0, col])

        # Coverage
        axes[1, col].set_ylim(0., 1.05)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        sns.scatterplot(
            data=df_plot,
            x="std",
            y="coverage",
            hue="learning_rate",
            style="clip_param",
            markers={"0.2": "o", "0.0": "X"},
            palette="deep",
            ax=axes[1, col],
            s=80,
            alpha=0.6,
            legend=False
        )
        if col == 0:
            axes[1, col].set_ylabel("Coverage")
        else:
            axes[1, col].set_ylabel(None)
        customize_axis(axes[1, col])

        # Max Fitness
        sns.scatterplot(
            data=df_plot,
            x="std",
            y="max_fitness",
            hue="learning_rate",
            style="clip_param",
            markers={"0.2": "o", "0.0": "X"},
            palette="deep",
            ax=axes[2, col],
            s=80,
            alpha=0.6,
            legend=False
        )
        if col == 0:
            axes[2, col].set_ylabel("Max Fitness")
        else:
            axes[2, col].set_ylabel(None)
        customize_axis(axes[2, col])

    # Extract handles and labels from the first subplot (top-left)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if axes[0,0].get_legend() is not None:
        axes[0, 0].get_legend().remove()

    # Create a single legend at the bottom center of the figure
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(labels), frameon=False)

    fig.align_ylabels(axes)
    fig.tight_layout()

    fig.savefig("final_tuning_pt2/output/plot_main.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    results_dir = Path("final_tuning_pt2/output/")
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)
    df = df[df["algo"].isin(ALGO_LIST)]
    #df = df[~df['learning_rate'].isin([0.3, 0.03, 0.0003])]
    #df = df[~df['no_epochs'].isin([1, 2, 4, 8, 16])]

    df = df[df["num_evaluations"] <= 1_001_400]

    plot(df)