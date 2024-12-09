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
    "A100": "GPU: A100",
    "L40S": "GPU: L40S",
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
    256,
    512,
    1024,
    2048,
    4096,
    #16384,
    #32768,
    #65536
    
]

ALGO_LIST = [
    #"dcg_me",
    #"dcg_me_gecco",
    #"pga_me",
    #"qd_pg",
    #"me",
    #"me_es",
    #"mcpg_me",
    #"memes",
    #"mcpg_me_fixed",
    "mcpg_me_32"
]

NEW_ALGO_LIST = [
    #"mcpg_me_",
    #"mcpg_me_2",
    #"mcpg_me_4",
    #"mcpg_me_8",
    #"mcpg_me_16",
    #"mcpg_me_32",
    #"mcpg_me_unif_0",
    #"mcpg_me_unif_05",
    #"mcpg_me_unif_1_cos_sim",
    #"mcpg_me_unif_1_not_cos_sim",
]
ALGO_DICT = {
    "dcg_me": "DCG-MAP-Elites-AI",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "mcpg_me_2": "2 epochs",
    "mcpg_me_4": "4 epochs",
    "mcpg_me_8": "8 epochs",
    "mcpg_me_16": "16 epochs",
    "mcpg_me_32": "MCPG-ME",
    "memes": "MEMES",
    "mcpg_me_orth_0_cos_sim": "MCPG-ME orth 0 cos_sim",
    "mcpg_me_orth_0_not_cos_sim": "MCPG-ME orth 0 not_cos_sim",
    "mcpg_me_orth_05": "MCPG-ME orth 0.5",
    "mcpg_me_unif_0": "MCPG-ME unif 0",
    "mcpg_me_unif_05": "MCPG-ME unif 0.5",
    "mcpg_me_unif_1_cos_sim": "MCPG-ME unif 1 cos_sim",
    "mcpg_me_unif_1_not_cos_sim": "MCPG-ME unif 1 not_cos_sim",
}

XLABEL = "Evaluations"


def filter(df_row):
    if df_row["algo"] == "mcpg_me_fixed":
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"]:
        #    return "mcpg_me_orth_0_cos_sim"
        
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"] and df_row["no_epochs"] == 2:
        #    return "mcpg_me_2"
        
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"] and df_row["no_epochs"] == 4:
        #    return "mcpg_me_4"
        
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"] and df_row["no_epochs"] == 8:
        #    return "mcpg_me_8"
        
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"] and df_row["no_epochs"] == 16:
        #    return "mcpg_me_16"
        
        if df_row["init"] == "orthogonal" and df_row["greedy"] == 0 and df_row["cos_sim"] and df_row["no_epochs"] == 32 and df_row["proportion_mutation_ga"] == 0.5 and df_row["clip_param"] == 0.2:
            return "mcpg_me_32"

        
        #if df_row["init"] == "orthogonal" and df_row["greedy"] == 0.5:
        #    return "mcpg_me_orth_05"
        
        #if df_row["init"] == "uniform" and df_row["greedy"] == 0:
        #    return "mcpg_me_unif_0"
        
        #if df_row["init"] == "uniform" and df_row["greedy"] == 0.5:
        #    return "mcpg_me_unif_05"
        
        #if df_row["init"] == "uniform" and df_row["greedy"] == 1 and df_row["cos_sim"]:
        #    return "mcpg_me_unif_1_cos_sim"
        
        #if df_row["init"] == "uniform" and df_row["greedy"] == 1 and not df_row["cos_sim"]:
        #    return "mcpg_me_unif_1_not_cos_sim"
            
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
    fig, axes = plt.subplots(nrows=len(ENV_LIST), ncols=2, sharex='col', figsize=(25, 15))



    # Define a suitable color palette
    color_palette = sns.color_palette("viridis", len(BATCH_LIST))

            
    #formatter = ScalarFormatter(useMathText=True)
    #formatter.set_scientific(True)
    #formatter.set_powerlimits((0, 0))

    

    
    x_label = "Batch Size"
    y_labels = ["QD Score after 5M evaluations", "Runtime (s) after 5M evaluations"]

    for col in range(2):
        for row, env in enumerate(ENV_LIST):
            ax = axes[row, col]
            
            # Set title for each subplot
            if row == 0:
                ax.set_title(f"{y_labels[col]} vs. {x_label}")
            
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
                    ax=ax,
                    palette=color_palette,
                    legend=False,
                    dodge=True
                )
                ax.set_ylabel(f"{ENV_DICT[env]}")
                #sax.yaxis.set_major_formatter(formatter)  # Scientific notation
            else:
                sns.barplot(
                    data=df_plot,
                    x="algo",
                    y="time",
                    hue="batch_size",
                    ax=ax,
                    palette=color_palette,
                    legend=False,
                    dodge=True
                )
                #ax.set_ylabel("Runtime (s)")
                ax.set_ylabel(None)


                # Set the x-axis labels with rotation for better visibility
                #ax.set_xticklabels([str(batch_size) for batch_size in BATCH_LIST], rotation=45, ha="right")
            ax.set_xticklabels([ALGO_DICT.get(algo, algo) for algo in df_plot['algo'].unique()], ha="center")
            # Set y-axis label and limits
            #ax.set_ylim(0.0)
            #ax.set_ylabel(None)
            ax.set_xlabel(None)
                
            # Customize the axis aesthetics
            customize_axis(ax)
        

        # Legend and final adjustments
        colors = sns.color_palette(palette=color_palette, n_colors=len(BATCH_LIST))  # Get a color palette with 3 distinct colors
        patches = [mlines.Line2D([], [], color=colors[i], label=str(batch_size), linewidth=2.2, linestyle='-') for i, batch_size in enumerate(BATCH_LIST)]
        fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)    
    
        #fig.legend(ax_.get_lines(), [str(batch_size) for batch_size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)
        fig.align_ylabels(axes[:, 0])
        fig.tight_layout()
        fig.savefig("tuning/output/plot_main_epochs_scal.png", bbox_inches="tight")
        plt.close()




if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/2
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("tuning/output/")
    #print(results_dir)
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_000_000]
    #df['env_'] = df.apply(filter_gpu_variants, axis=1)
#    df = df.loc[
#    (df['algo'] != "mcpg_me") | 
#    ((df['algo'] == "mcpg_me") & (df['proportion_mutation_ga'] == 1))
#]
    #df['algo_'] = df.apply(filter, axis=1)
    
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()

    df_last_iteration = df.loc[idx]

    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score', 'batch_size']]
    summary_df = summary_df[summary_df["batch_size"].isin(BATCH_LIST)]
    
    #df['env_'] = df.apply(filter_gpu_variants, axis=1)

    # Plot
    #plot(df)
    
    #plot_(df)
    
    plot__(summary_df)