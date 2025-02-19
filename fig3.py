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
    #"humanoid_omni",
    "anttrap_omni_250",
    "walker2d_uni_250",
    #"walker2d_uni_1000",
    #"halfcheetah_uni",
    "ant_uni_250",
    "hopper_uni_250",
    #"ant_uni_1000",
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
    "mcpg_me",
    "dcg_me",
    "pga_me",
    "me",
    #"memes",
]


ALGO_DICT = {
    "mcpg_me_0": "MCPG-ME 0% GA",
    "mcpg_me_0.25": "MCPG-ME 25% GA",
    "mcpg_me_0.5": "MCPG-ME 50% GA",
    "mcpg_me_0.75": "MCPG-ME 75% GA",
    "mcpg_me_1": "MCPG-ME 100% GA",
    #"dcg_me": "DCRL-AI-only",
    "dcg_me": "DCRL-ME",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-ME",
    "qd_pg": "QD-PG",
    "me": "ME",
    "me_es": "MAP-Elites-ES",
    "mcpg_me": "ASCII-ME",
    "memes": "MEMES",
    "mcpg_me_no_normalizer": "Ablation 1",
    "mcpg_me_no_baseline": "Ablation 2",
    "mcpg_me_no_ppo_loss": "Ablation 3",
    "mcpg_me_": "MCPG-ME old",
    "mcpg_me_fixed": "MCPG-ME fixed",
    "mcpg_me_orth_0_cos_sim": "MCPG-ME orth 0 cos_sim",
    "mcpg_me_orth_0_not_cos_sim": "MCPG-ME orth 0 not_cos_sim",
    "mcpg_me_orth_05": "MCPG-ME orth 0.5",
    "mcpg_me_unif_0": "MCPG-ME unif 0",
    "mcpg_me_unif_05": "MCPG-ME unif 0.5",
    "mcpg_me_unif_1_cos_sim": "MCPG-ME unif 1 cos_sim",
    "mcpg_me_unif_1_not_cos_sim": "MCPG-ME unif 1 not_cos_sim",
    "mcpg_me_25": "MCPG-ME | 25% GA",
    "mcpg_me_50": "MCPG-ME | 50% GA",
    "mcpg_me_75": "MCPG-ME | 75% GA",
    "mcpg_me_0": "MCPG-ME | 0% GA",
    "mcpg_me_no_clipping": "MCPG-ME no clip",
    "mcpg_me_normal": "MCPG-ME with clip",
    "mcpg_me_epoch_32_batch_512" : "MCPG-ME",
    "mcpg_me_clip_1": "clip 1",
    "mcpg_me_clip_2": "clip 2",
    "mcpg_me_clip_3": "clip 3",
    "mcpg_me_no_clip_05": "no clip 0.5",
    "mcpg_me_no_clip_1": "no clip 1",
}

EMITTER_LIST = {
    #"dcg_me" : ["ga_offspring_added", "qpg_offspring_added", "ai_offspring_added"],
    #"pga_me" : ["ga_offspring_added", "qpg_offspring_added", "ai_offspring_added"],
    "mcpg_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "dcg_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    "pga_me": ["ga_offspring_added", "qpg_ai_offspring_added"],
    #"me": ["ga_offspring_added"],    
}
EMITTER_DICT = {
    "ga_offspring_added": "Iso+LineDD",
    #"qpg_offspring_added": "PG",
    #"dpg_offspring_added": "DPG",
    #"es_offspring_added": "ES",
    #"ai_offspring_added": "AI",
    "qpg_ai_offspring_added": "PG + AI",
}

XLABEL = "Evaluations"

def filter(df_row):
    if df_row["algo"] == "pga_me":
        if df_row["batch_size"] != 1024:
            return 
    if df_row["algo"] == "dcg_me":
        if df_row["batch_size"] != 2048:
            return 

    if df_row["algo"] == "mcpg_me":
        if df_row["batch_size"] != 4096 or df_row["proportion_mutation_ga"] != 0.5 or df_row["greedy"] != 0:
            return 
        
    if df_row["algo"] == "me":
        if df_row["batch_size"] != 8192:
            return 
    

            
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
    fig.savefig("fig1/output/fig3.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("fig1/output/")
        
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
    #df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    #df['algo_'] = df.apply(filter, axis=1)
    # Plot
    plot(df)
    