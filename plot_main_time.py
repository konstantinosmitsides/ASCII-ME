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
    'mcpg_me',
    'mcpg_me_ga_0_greedy_0',
    'mcpg_me_ga_25_greedy_0',
    'mcpg_me_ga_75_greedy_0',
    'mcpg_me_ga_1_greedy_0',
    # 'mcpg_me_1',
    "pga_me",
]


ALGO_DICT = {
    "dcg_me": "DCRL",
    "dcg_me_": "DCRL",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "mcpg_me": "MCPG-ME 50% GA (MCPG-ME)",
    "mcpg_me_25_greedy_0": "MCPG-ME 25% GA",
    "mcpg_me_75_greedy_0": "MCPG-ME 75% GA",
    "mcpg_me_1_greedy_0": "MCPG-ME 100% GA",
    "memes": "MEMES",
    "ppga" : "PPGA",
    "mcpg_only": "MCPG-Only",
    "mcpg_only_05": "MCPG-Only (0.5)",
    "mcpg_only_1": "MCPG-Only (1)",
    "mcpg_me_05": "MCPG-ME (0.5)",
    "mcpg_me_1": "MCPG-ME (1)",
}


BATCH_LIST = [
    512,
    1024,
    2048,
    4096,
]

ALGO_DICT = {
    "dcg_me": "DCRL",
    "dcg_me_": "DCRL",
    "dcg_me_gecco": "DCG-MAP-Elites GECCO",
    "pga_me": "PGA-MAP-Elites",
    "qd_pg": "QD-PG",
    "me": "MAP-Elites",
    "me_es": "MAP-Elites-ES",
    "mcpg_me": "MCPG-ME 50% GA (MCPG-ME)",
    "mcpg_me_25_greedy_0": "MCPG-ME 25% GA",
    "mcpg_me_75_greedy_0": "MCPG-ME 75% GA",
    "mcpg_me_1_greedy_0": "MCPG-ME 100% GA",
    "memes": "MEMES",
    "ppga" : "PPGA",
    "mcpg_only": "MCPG-Only",
    "mcpg_only_05": "MCPG-Only (0.5)",
    "mcpg_only_1": "MCPG-Only (1)",
    "mcpg_me_05": "MCPG-ME (0.5)",
    "mcpg_me_1": "MCPG-ME (1)",
}

XLABEL = "Time (s)"


def filter(df_row):
    if df_row["algo"] == "pga_me":
        if df_row["batch_size"] != 1024:
            return 
    if df_row["algo"] == "ppga":
        if df_row["batch_size"] != 6000:
            return 
    if df_row["algo"] == "memes":
        if df_row["batch_size"] != 8192:
            return 
    if df_row["algo"] == "mcpg_me":
        if df_row["batch_size"] != 4096:
            return 
    
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 0:
            return "mcpg_me_ga_0_greedy_0"
        
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.25 and df_row["greedy"] == 0:
            return "mcpg_me_ga_25_greedy_0"

    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.75 and df_row["greedy"] == 0:
            return "mcpg_me_ga_75_greedy_0"
        
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 1 and df_row["greedy"] == 0:
            return "mcpg_me_ga_1_greedy_0"


    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 0.5:
            return "mcpg_me_ga_0_greedy_05"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0 and df_row["greedy"] == 1:
            return "mcpg_me_ga_0_greedy_1"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.5 and df_row["greedy"] == 0.5:
            return "mcpg_me_ga_05_greedy_05"
    if df_row["algo"] == "mcpg_me":
        if df_row["proportion_mutation_ga"] == 0.5 and df_row["greedy"] == 1:
            return "mcpg_me_ga_05_greedy_1"
    return df_row["algo"]

def simple_moving_average(data, window_size):
    """Calculates the simple moving average over a specified window size."""
    return data.rolling(window=window_size, center=False).mean()

# Function to concatenate data and apply moving average
def aggregate_and_smooth(df, window_size=100):
    # Empty DataFrame to store results
    smoothed_data = pd.DataFrame()
    
    # Loop over each unique environment and algorithm combination
    for (env, algo), group in df.groupby(['env', 'algo']):
        # Concatenate the relevant columns for all runs
        aggregated_qd_score = pd.concat([group.loc[group['run'] == run, 'qd_score'] for run in group['run'].unique()])
        aggregated_max_fitness = pd.concat([group.loc[group['run'] == run, 'max_fitness'] for run in group['run'].unique()])
        aggregated_coverage = pd.concat([group.loc[group['run'] == run, 'coverage'] for run in group['run'].unique()])
        
        # Calculate moving averages
        qd_score_sma = simple_moving_average(aggregated_qd_score, window_size)
        max_fitness_sma = simple_moving_average(aggregated_max_fitness, window_size)
        coverage_sma = simple_moving_average(aggregated_coverage, window_size)
        
        # Collect results in a DataFrame
        result = pd.DataFrame({
            'env': env,
            'algo': algo,
            'qd_score': qd_score_sma,
            'max_fitness': max_fitness_sma,
            'coverage_sma': coverage_sma
        })
        smoothed_data = pd.concat([smoothed_data, result])
    
    return smoothed_data.reset_index(drop=True)

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
    fig.savefig("fig4/output/GA_ablation.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("fig4/output/")
    #print(results_dir)
    EPISODE_LENGTH = 250
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    #df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_005_00]

    
    
    # Get the median time for each (env, algo)

# Using .apply() to customize the index selection per group
#    idx = df.groupby(["env", "algo", "run"]).apply(
#    lambda x: 60 if x['algo'].iloc[0] == 'memes' else min(x['iteration'].idxmax(), x.index.min() + 1999, x.index.max())
#)
    '''
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    time_median = df_last_iteration.groupby(["env", "algo"])["time"].median()
    df_last_iteration = df_last_iteration.join(time_median, on=["env", "algo"], rsuffix="_median")
    
    # Get the difference between the time and the median for each run
    df_last_iteration["time_diff"] = df_last_iteration["time"] - df_last_iteration["time_median"]
    
    # Get the most representative run for each (env, algo)
    idx = df_last_iteration.groupby(['env', 'algo'])['time_diff'].idxmin()
    runs = df_last_iteration.loc[idx][["env", "algo", "run"]]


    merged_df = pd.merge(runs, df, on=['env', 'algo', 'run'], how='inner')

    
    # Select every kth element
    #k = 500  # Adjust this value to your needs
    #df = df.iloc[::k, :]

    # Plot
    plot(merged_df)
    '''
    
    '''
    df['iteration_block'] = (df['iteration'] - 1) // 10  # Blocks of 10 iterations

    # Step 2: Group by 'env', 'algo', 'iteration_block', and aggregate the required metrics
    aggregated_df = df.groupby(['env', 'algo', 'iteration_block']).agg({
        'qd_score': 'mean',
        'coverage': 'mean',
        'max_fitness': 'mean',
        'time': 'mean'
    }).reset_index()

    # Step 3: Since we want the mean of means across all 10 runs per block, we need another aggregation step
    final_df = aggregated_df.groupby(['env', 'algo', 'iteration_block']).mean().reset_index()

    # Printing the result to check
    print(final_df.head())
    '''
    
    df['iteration_block'] = (df['iteration'] - 1) // 10

    # Step 2: Group by 'env', 'algo', 'run', 'iteration_block' and calculate the means for each metric per run
    metrics = ['qd_score', 'coverage', 'max_fitness']
    grouped = df.groupby(['env', 'algo', 'run', 'iteration_block'])
    aggregated_per_run = grouped[metrics].mean().reset_index()

    # Calculate the average time across all runs for each iteration block in each env and algo
    average_time = df.groupby(['env', 'algo', 'iteration_block'])['time'].mean().reset_index()

    # Merge the average time back with the per-run metrics
    final_df = pd.merge(aggregated_per_run, average_time, on=['env', 'algo', 'iteration_block'])

    # Sorting for better structure and resetting index for cleanliness
    final_df = final_df.sort_values(by=['env', 'algo', 'run', 'iteration_block']).reset_index(drop=True)

    # Plot
    plot(final_df)