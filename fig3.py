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

BATCH_LIST = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,    
]

ALGO_LIST = [
    'ascii_me',
    "dcrl_me",
    "pga_me",
    "me",
    "memes",
]


ALGO_DICT = {
    "dcrl_me": "DCRL-ME",
    "pga_me": "PGA-ME",
    "ascii_me": "ASCII-ME",
    "memes": "MEMES",
    "ppga" : "PPGA",
    "me": "ME",
}

XLABEL = "Evaluations"


def filter(df_row):
        
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

    # Remove ticks
    # ax.tick_params(axis="y", length=0)

    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax

def plot__(summary_df):
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=2, sharex='col', figsize=(25, 15 * 0.7))



    # Define a suitable color palette
    color_palette = sns.color_palette("viridis", len(BATCH_LIST))

            
    #formatter = ScalarFormatter(useMathText=True)
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((0, 0))

    

    
    x_labels = ["Batch Size    (the higher the better)", "Batch Size    (the lower the better)"]
    y_labels = ["QD Score after 1M evaluations", "Runtime (s) after 1M evaluations"]

    for col in range(2):
        for row, env in enumerate(ENV_LIST):
            ax = axes[row, col]
            
            # Set title for each subplot
            if row == 0:
                ax.set_title(f"{y_labels[col]} vs. {x_labels[col]}")
            
            # Formatter for the x-axis
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.xaxis.get_major_formatter().set_scientific(True)
            ax.xaxis.get_major_formatter().set_powerlimits((0, 0))

            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.yaxis.get_major_formatter().set_scientific(True)
            ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

            # Get df for the current env
            df_plot = summary_df[summary_df["env"] == env]
            
            if col == 0:
                sns.barplot(
                    data=df_plot,
                    x="algo",
                    y="qd_score",
                    hue="batch_size",
                    hue_order=BATCH_LIST,
                    estimator=np.median,
                    errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                    ax=ax,
                    palette=color_palette,
                    legend=False,
                    dodge=True,
                    order=ALGO_LIST,
                )
                ax.set_ylabel(f"{ENV_DICT[env]}")
                #sax.yaxis.set_major_formatter(formatter)  # Scientific notation
            else:
                sns.barplot(
                    data=df_plot,
                    x="algo",
                    y="time",
                    hue="batch_size",
                    hue_order=BATCH_LIST,
                    estimator=np.median,
                    errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                    ax=ax,
                    palette=color_palette,
                    legend=False,
                    dodge=True,
                    order=ALGO_LIST,
                )
                #ax.set_ylabel("Runtime (s)")
                ax.set_ylabel(None)


                # Set the x-axis labels with rotation for better visibility
                #ax.set_xticklabels([str(batch_size) for batch_size in BATCH_LIST], rotation=45, ha="right")
            #ax.set_xticklabels([ALGO_DICT.get(algo, algo) for algo in df_plot['algo'].unique()], ha="center")
            ax.set_xticklabels([ALGO_DICT.get(algo, algo) for algo in ALGO_LIST], ha="center")
            # Set y-axis label and limits
            #ax.set_ylim(0.0)
            #ax.set_ylabel(None)
            ax.set_xlabel(None)
                
            # Customize the axis aesthetics
            customize_axis(ax)
            ax.set_axisbelow(True)
        

        # Legend and final adjustments
        colors = sns.color_palette(palette=color_palette, n_colors=len(BATCH_LIST))  # Get a color palette with 3 distinct colors
        patches = [mlines.Line2D([], [], color=colors[i], label=str(batch_size), linewidth=2.2, linestyle='-') for i, batch_size in enumerate(BATCH_LIST)]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)  
        #fig.text(0.1, 0.02, "Batch Size", ha='right', va='center', fontsize=12)  
    
        #fig.legend(ax_.get_lines(), [str(batch_size) for batch_size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        fig.savefig("output/fig3.png", bbox_inches="tight")
        plt.close()




if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/2
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("output/")
    #print(results_dir)
    
    df = get_df(results_dir)

    # Filter
    df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_005_000]

    
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()

    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score', 'batch_size']]
    summary_df = summary_df[summary_df["batch_size"].isin(BATCH_LIST)]
    
    
    plot__(summary_df)