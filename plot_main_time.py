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
ALGO_LIST = [
    "mcpg_me",
    #"mcpg_me_no_normalizer",
    #"mcpg_me_no_baseline",
    #"mcpg_me_no_ppo_loss",
    "dcg_me",
    #"dcg_me_gecco",
    "pga_me",
    #"qd_pg",
    #"me_es",
    "memes",
    "me",
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
    "mcpg_me_no_normalizer": "MCPG-ME no normalizer",
    "mcpg_me_no_baseline": "MCPG-ME no baseline",
    "mcpg_me_no_ppo_loss": "MCPG-ME no PPO loss",
}

XLABEL = "Time (s)"


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
            x="time",
            y="qd_score",
            hue="algo",
            hue_order=ALGO_LIST,
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
            y="coverage",
            hue="algo",
            hue_order=ALGO_LIST,
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
            x="time",
            y="max_fitness",
            hue="algo",
            hue_order=ALGO_LIST,
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
    fig.savefig("data_time_efficiency/output/plot_main_time.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("data_time_efficiency/output/")
    #print(results_dir)
    EPISODE_LENGTH = 250
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    
    
    # Get the median time for each (env, algo)

# Using .apply() to customize the index selection per group
#    idx = df.groupby(["env", "algo", "run"]).apply(
#    lambda x: 60 if x['algo'].iloc[0] == 'memes' else min(x['iteration'].idxmax(), x.index.min() + 1999, x.index.max())
#)
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    time_median = df_last_iteration.groupby(["env", "algo"])["time"].median()
    df_last_iteration = df_last_iteration.join(time_median, on=["env", "algo"], rsuffix="_median")
    
    # Get the difference between the time and the median for each run
    df_last_iteration["time_diff"] = df_last_iteration["time"] - df_last_iteration["time_median"]
    
    # Get the most representative run for each (env, algo)
    idx = df_last_iteration.groupby(['env', 'algo'])['time_diff'].idxmin()
    runs = df_last_iteration.loc[idx][["env", "algo", "run"]]
    
    '''
    import os
    import shutil

    # Define the root directory for plotting
    root_dir = './for_plotting_time'

    # Ensure the root directory exists
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Iterate through each row in the runs DataFrame
    for index, row in runs.iterrows():
        # Construct the path for the environment directory
        env_dir = os.path.join(root_dir, row['env'])
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)

        # Construct the path for the algorithm directory within the environment directory
        algo_dir = os.path.join(env_dir, row['algo'])
        if not os.path.exists(algo_dir):
            os.makedirs(algo_dir)

        # Construct the path for the run directory within the algorithm directory
        run_dir = os.path.join(algo_dir, f"run_{row['run']}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

    # Optional: Move or copy files to the run directory
    # You can use shutil.copy(src_file, run_dir) or shutil.move(src_file, run_dir)
    # depending on whether you want to copy or move files
    '''

    merged_df = pd.merge(runs, df, on=['env', 'algo', 'run'], how='inner')

    
    # Select every kth element
    #k = 500  # Adjust this value to your needs
    #df = df.iloc[::k, :]

    # Plot
    plot(merged_df)