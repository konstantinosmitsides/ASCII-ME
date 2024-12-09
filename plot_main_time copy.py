import sys
sys.path.append("/project/")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
from matplotlib.patches import Rectangle, Patch

from utils import get_df


# Define environment and algorithm names
ENV_LIST = [
    "ant_omni_250",
    "anttrap_omni_250",
    # "humanoid_omni",
    "walker2d_uni_250",
    # "walker2d_uni_1000",
    # "halfcheetah_uni",
    "ant_uni_250",
    # "ant_uni_1000",
    "hopper_uni_250",
    # "hopper_uni_1000",
    # "humanoid_uni",
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
    # "mcpg_me_no_normalizer",
    # "mcpg_me_no_baseline",
    # "mcpg_me_no_ppo_loss",
    # "mcpg_me_epoch_32_batch_512",
    # "mcpg_me_epoch_32_batch_1024",
    # "mcpg_me_epoch_32_batch_2048",
    # "mcpg_me_epoch_32_batch_4096",
    "dcg_me",
    # "dcg_me_",
    # "dcg_me_gecco",
    "pga_me",
    # "qd_pg",
    # "me_es",
    "memes",
    "me",
]

ALGO_DICT = {
    "dcg_me": "DCRL",
    "dcg_me_": "DCRL",
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
    "mcpg_me_epoch_32_batch_512": "MCPG-ME",
    "mcpg_me_epoch_32_batch_1024": "epoch 32 batch 1024",
    "mcpg_me_epoch_32_batch_2048": "epoch 32 batch 2048",
    "mcpg_me_epoch_32_batch_4096": "epoch 32 batch 4096",
}

XLABEL = "Runtime (s)"


def filter(df_row):
    if df_row["algo"] == "mcpg_me_fixed":
        if (
            df_row["init"] == "orthogonal"
            and df_row["greedy"] == 0
            and df_row["cos_sim"]
            and df_row["no_epochs"] == 32
            and df_row["batch_size"] == 512
            and df_row["proportion_mutation_ga"] == 0.5
            and df_row["clipping"] == 0.2
        ):
            return "mcpg_me"
    return df_row["algo"]


def customize_axis(ax):
    # Remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add grid
    ax.grid(which="major", axis="x", color="0.9")
    return ax


def plot(df_summary):
    # List of metrics to plot
    metrics = ["qd_score", "coverage", "max_fitness"]
    num_metrics = len(metrics)
    num_envs = len(ENV_LIST)

    # Create color palette
    color_palette = sns.color_palette(None, len(ALGO_LIST))
    algo_colors = dict(zip(ALGO_LIST, color_palette))

    # Create subplots
    fig, axes = plt.subplots(
        nrows=num_metrics,
        ncols=num_envs,
        figsize=(5 * num_envs, 4 * num_metrics),
        sharex=True,
    )

    for col, env in enumerate(ENV_LIST):
        for row, metric in enumerate(metrics):
            ax = axes[row, col] if num_metrics > 1 else axes[col]

            # Get data for this environment and metric
            df_env = df_summary[df_summary["env"] == env]

            # Prepare data
            algos = df_env["algo"]
            # Ensure consistent algorithm ordering
            algos_ordered = [algo for algo in ALGO_LIST if algo in algos.values]
            df_plot = df_env.set_index("algo").loc[algos_ordered].reset_index()

            # Sort df_plot by time_median ascending to plot shorter time bars last (on top)
            df_plot = df_plot.sort_values(by="time_median", ascending=False).reset_index(drop=True)

            # Keep track of previously plotted bars for overlap checking
            plotted_bars = []

            for idx, row_data in df_plot.iterrows():
                algo = row_data["algo"]
                time_median = row_data["time_median"]
                time_error = [
                    row_data["time_median"] - row_data["time_q25"],
                    row_data["time_q75"] - row_data["time_median"],
                ]
                metric_median = row_data[f"{metric}_median"]
                metric_q25 = row_data[f"{metric}_q25"]
                metric_q75 = row_data[f"{metric}_q75"]

                # Calculate bar position and height (IQR)
                y_bottom = metric_q25
                y_top = metric_q75
                y_height = y_top - y_bottom

                # Check for overlap with previously plotted bars
                overlap = False
                for bar in plotted_bars:
                    # bar = (y_bottom, y_top)
                    if (y_bottom < bar[1]) and (y_top > bar[0]):
                        overlap = True
                        break

                # Adjust alpha based on overlap
                if overlap:
                    bar_alpha = 1.0  # More opaque for overlapping bars
                else:
                    bar_alpha = 0.7  # Default opacity

                # Plot the bar as a rectangle
                rect = Rectangle(
                    (0, y_bottom),  # (x, y) bottom-left corner
                    width=time_median,  # width along x-axis (runtime)
                    height=y_height,  # height along y-axis (IQR)
                    color=algo_colors[algo],
                    alpha=bar_alpha,
                    edgecolor="k",
                    label=ALGO_DICT.get(algo, algo),
                    zorder=idx,  # Later items have higher zorder
                )
                ax.add_patch(rect)

                # Add error bars for runtime
                ax.errorbar(
                    x=time_median,
                    y=metric_median,
                    xerr=[[time_error[0]], [time_error[1]]],
                    fmt="none",
                    ecolor="black",
                    capsize=5,
                    zorder=idx + 1,  # Ensure error bars are on top
                )

                # Add this bar's y-range to the list of plotted bars
                plotted_bars.append((y_bottom, y_top))

            # Set labels and title
            if row == 0:
                ax.set_title(ENV_DICT[env])
            if row == num_metrics - 1:
                ax.set_xlabel(XLABEL)
            if col == 0:
                ax.set_ylabel(metric.replace("_", " ").capitalize())
            else:
                ax.set_ylabel("")

            # Set y-limits to accommodate the IQR bars
            all_metrics = df_plot[[f"{metric}_q25", f"{metric}_q75"]].values.flatten()
            y_min, y_max = all_metrics.min(), all_metrics.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

            # Customize axis
            customize_axis(ax)

            # Adjust x-axis ticks to have more splits
            # Determine the maximum runtime to set the x-axis limit
            max_time = df_plot["time_median"].max()
            ax.set_xlim(0, max_time + 0.1 * max_time)  # Add some padding

            # Set the number of x-axis ticks
            num_ticks = 5  # Adjust this number as needed
            ax.locator_params(axis='x', nbins=num_ticks)

            # Use ScalarFormatter for scientific notation if needed
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

            # Adjust y-axis formatting for specific subplots
            # Rows 0 and 2 correspond to qd_score and max_fitness (indexes start at 0)
            # Columns 2, 3, 4 correspond to columns 3, 4, 5 (indexes start at 0)
            if (row == 0 or row == 2) and (col >= 2):
                # Use ScalarFormatter for y-axis
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
                # Set the number of y-axis ticks
                ax.locator_params(axis='y', nbins=5)

    # Adjust layout and add legend at the bottom center
    # Create legend handles and labels
    legend_patches = [
        Patch(color=algo_colors[algo], label=ALGO_DICT.get(algo, algo))
        for algo in ALGO_LIST
        if algo in algo_colors
    ]

    # Add legend to the figure
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(legend_patches),
        frameon=False,
    )

    plt.tight_layout()
    # Save plot
    plt.savefig("data_time_efficiency/output/plot_main_time.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid Type 3 fonts in matplotlib
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("data_time_efficiency/output/")
    EPISODE_LENGTH = 250
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df["algo"] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]

    # Extract final performance metrics and runtime
    df_final = df.groupby(["env", "algo", "run"]).last().reset_index()

    # Function to compute summary statistics
    def compute_summary(df):
        def q25(x):
            return np.quantile(x, 0.25)

        def q75(x):
            return np.quantile(x, 0.75)

        df_summary = (
            df.groupby(["env", "algo"])
            .agg(
                qd_score_median=("qd_score", "median"),
                qd_score_q25=("qd_score", q25),
                qd_score_q75=("qd_score", q75),
                coverage_median=("coverage", "median"),
                coverage_q25=("coverage", q25),
                coverage_q75=("coverage", q75),
                max_fitness_median=("max_fitness", "median"),
                max_fitness_q25=("max_fitness", q25),
                max_fitness_q75=("max_fitness", q75),
                time_median=("time", "median"),
                time_q25=("time", q25),
                time_q75=("time", q75),
            )
            .reset_index()
        )
        return df_summary

    df_summary = compute_summary(df_final)

    # Plot
    plot(df_summary)