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
    "me",
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
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=3, sharex='col', figsize=(25, 15))

    # Metrics and axes labels
    x_metrics = ['num_evaluations', 'iteration']#, 'time']
    y_metrics = ['qd_score', 'qd_score']#, 'qd_score']
    x_labels = ['Evaluations', 'Iterations']#, 'Runtime (s)']
    y_labels = ['QD score', 'QD score']#, 'QD score']

    # Define a suitable color palette
    color_palette = sns.color_palette("Set2", len(BATCH_LIST))

    for col, (x_metric, y_metric, x_label, y_label) in enumerate(zip(x_metrics, y_metrics, x_labels, y_labels)):
        for row, env in enumerate(ENV_LIST):
            ax = axes[row, col]

            # Set titles and labels
            if row == 0:
                ax.set_title(f"{y_label} vs. {x_label}")
            if col == 0:
                ax.set_ylabel(ENV_DICT[env])
            if row == len(ENV_LIST) - 1:
                ax.set_xlabel(x_label)

            # Apply formatter
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
            

            # Plotting
            sns.lineplot(
                data=df[df["env"] == env],
                x=x_metric,
                y=y_metric,
                hue="batch_size",
                hue_order=BATCH_LIST,
                palette=color_palette,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=ax
            )
            
            ax.set_xlabel(None)
            if col != 0:
                ax.set_ylabel(None)
            customize_axis(ax)
            
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    
    
    idx = df.groupby(["env", "algo", "batch_size", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', "batch_size", 'time']]
    
    
    col = 2
    y_label = "Runtime (s)"
    x_label = "Batch size"
    for row, env in enumerate(ENV_LIST):
        ax = axes[row, col]
        
        # Set title for each subplot
        #ax.set_title(ENV_DICT[env])
        if row == 0:
            ax.set_title(f"{y_label} vs. {x_label}")
        ax.xaxis.set_major_formatter(formatter)

        
        # Get df for the current env
        df_plot = summary_df[summary_df["env"] == env]
        ax.yaxis.set_major_formatter(formatter)
        
        # Create boxplot for 'time' of each algorithm
        #sns.boxplot(
        #    data=df_plot,
        #    x="batch_size",
        #    y="time",
        #    hue="batch_size",
        #    hue_order=BATCH_LIST,
        #    order=BATCH_LIST,
        #    palette=color_palette,
        #    legend=False,
        #    ax=ax,
        #)
        sns.barplot(
            data=df_plot,
            x="batch_size",
            y="time",
            estimator=np.median,  
            errorbar=None,
            ax=ax,
            hue="batch_size",
            hue_order=BATCH_LIST,  
            order=BATCH_LIST,  
            palette=color_palette,
            legend=False,  
        )
        
        
        # Set the x-axis labels (Algorithm names) with rotation for better visibility
        ax.set_xticklabels([str(batch_size) for batch_size in BATCH_LIST], rotation=45, ha="right")
        
        # Label the y-axis with 'Time (s)' for the first subplot only
        ax.set_ylim(0.0)
        ax.set_xlabel(None)
        ax.set_ylabel(None)

        # Customize the axis aesthetics
        customize_axis(ax)
    

    # Legend and final adjustments
    colors = sns.color_palette(palette=color_palette, n_colors=len(BATCH_LIST))  # Get a color palette with 3 distinct colors
    patches = [mlines.Line2D([], [], color=colors[i], label=str(batch_size), linewidth=2.2, linestyle='-') for i, batch_size in enumerate(BATCH_LIST)]
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)    
 
    #fig.legend(ax_.get_lines(), [str(batch_size) for batch_size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)
    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
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
                axes[row, col].set_title(f"{y_label} vs {x_axis}")

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
    df = df[df["num_evaluations"] <= 5_000_000]
    
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score']]

    # Plot
    #plot(df)
    
    #plot_(df)
    
    plot__(df)