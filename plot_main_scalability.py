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
    #"anttrap_omni_250",
    #"humanoid_omni",
    "walker2d_uni_250",
    #"walker2d_uni_1000",
    #"halfcheetah_uni",
    "ant_uni_250",
    #"ant_uni_1000",
    #"hopper_uni_250",
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

BATCH_LIST = [
    1024,
    4096,
    16384,
    32768,
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


def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax

def plot__(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=3, sharex='col', figsize=(25, 15))  # Adjusted subplot dimensions

    # Metrics to plot, adjust according to your data
    metrics = ['qd_score', 'qd_score', 'qd_score']
    x_axes = ['num_evaluations', 'time', 'iteration']
    y_labels = ['QD Score', 'QD Score', 'QD Score']

    # Define a suitable color palette
    color_palette = sns.color_palette("Set2", len(BATCH_LIST))  # Change the palette if needed

    # Loop over each metric
    for col, (metric, x_axis, y_label) in enumerate(zip(metrics, x_axes, y_labels)):
        # Create formatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        for row, env in enumerate(ENV_LIST):
            # Set title for each column in the first row
            if row == 0:
                axes[row, col].set_title(f"{y_label} vs. {x_axis}")

            # Get df for the current environment
            df_plot = df[df["env"] == env]

            # Plotting
            ax = sns.lineplot(
                data=df_plot,
                x=x_axis,
                y=metric,
                hue="batch_size",
                hue_order=BATCH_LIST,
                palette=color_palette,  # Use the defined color palette
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[row, col],
            )

            # Set the y-axis label for the first column
            if col == 0:
                axes[row, col].set_ylabel(ENV_DICT[env])

            # Customize axis formatting
            axes[row, col].yaxis.set_major_formatter(formatter if metric != 'coverage' else PercentFormatter(1))
            customize_axis(axes[row, col])

    # Legend handling for the whole figure
    fig.legend(ax.get_lines(), [str(batch_size) for batch_size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)
    # Aesthetic and layout adjustments
    fig.align_ylabels(axes[:, 0])  # align y labels in the first column
    fig.tight_layout(rect=[0, 0.03, 1, 1])  # Adjust the rectangle in tight_layout

    # Save plot
    fig.savefig("scalability/output/plot_main.pdf", bbox_inches="tight")
    plt.close()


def plot_(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=3, sharex='col', figsize=(25, 15))  # Adjusted subplot dimensions

    # Metrics to plot, adjust according to your data
    metrics = ['qd_score', 'qd_score', 'qd_score']
    x_axes = ['num_evaluations', 'time', 'iteration']
    y_labels = ['QD Score', 'QD Score', 'QD Score']

    # Loop over each metric
    for col, (metric, x_axis, y_label) in enumerate(zip(metrics, x_axes, y_labels)):
        # Create formatter
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))

        for row, env in enumerate(ENV_LIST):
            # Set title for each column in the first row
            if row == 0:
                axes[row, col].set_title(f"{y_label} vs. {x_axis}")

            # Get df for the current environment
            df_plot = df[df["env"] == env]

            # Plotting
            ax = sns.lineplot(
                df_plot,
                x=x_axis,
                y=metric,
                hue="batch_size",
                hue_order=BATCH_LIST,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[row, col],
            )

            # Set the y-axis label for the first column
            if col == 0:
                axes[row, col].set_ylabel(ENV_DICT[env])

            # Customize axis formatting
            axes[row, col].yaxis.set_major_formatter(formatter if metric != 'coverage' else PercentFormatter(1))
            customize_axis(axes[row, col])

    # Adjust legend for the whole figure
    fig.legend(ax.get_lines(), [str(batch_size) for batch_size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)

    # Aesthetic and layout adjustments
    fig.align_ylabels(axes[:, 0])  # align y labels in the first column
    fig.tight_layout()

    # Save plot
    fig.savefig("scalability/output/plot_main.pdf", bbox_inches="tight")
    plt.close()

def plot(df):
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 10))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    for col, env in enumerate(ENV_LIST):
        print(env)

        # Set title for the column
        axes[0, col].set_title(ENV_DICT[env])

        # Set the x label and formatter for the column
        axes[2, col].set_xlabel(XLABEL)
        axes[2, col].xaxis.set_major_formatter(formatter)

        # Get df for the current env
        df_plot = df[df["env"] == env]

        # QD score
        axes[0, col].yaxis.set_major_formatter(formatter)

        sns.lineplot(
            df_plot,
            x="num_evaluations",
            y="qd_score",
            hue="batch_size",
            hue_order=BATCH_LIST,
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
            x="time",
            y="qd_score",
            hue="batch_size",
            hue_order=BATCH_LIST,
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[1, col],
        )

        if col == 0:
            axes[1, col].set_ylabel("Coverage")
        else:
            axes[1, col].set_ylabel(None)

        # Customize axis
        customize_axis(axes[1, col])

        # Max fitness
        ax = sns.lineplot(  # store ax for legend
            df_plot,
            x="iteration",
            y="qd_score",
            hue="batch_size",
            hue_order=BATCH_LIST,
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
    fig.legend(ax.get_lines(), [ALGO_DICT[algo] for algo in ALGO_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(ALGO_LIST), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("scalability/output/plot_main.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("scalability/output/")
    #print(results_dir)
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 10_000_000]

    # Plot
    #plot(df)
    
    #plot_(df)
    
    plot__(df)