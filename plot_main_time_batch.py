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
    #"ant_omni_250",
    #"anttrap_omni_250",
    #"humanoid_omni",
    #"walker2d_uni_250",
    "walker2d_uni_1000",
    #"halfcheetah_uni",
    #"ant_uni_250",
    "ant_uni_1000",
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
ALGO_LIST = [
    "mcpg_me",
    #"mcpg_me_no_normalizer",
    #"mcpg_me_no_baseline",
    #"mcpg_me_no_ppo_loss",
    #"dcg_me",
    #"dcg_me_gecco",
    #"pga_me",
    #"qd_pg",
    #"me_es",
    #"memes",
    #"me",
]

BATCH_LIST = [
    512,
    1024,
    2048,
    4096,
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
    
    color_palette = sns.color_palette("Set2", len(BATCH_LIST))
    

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
            hue="batch_size",
            hue_order=BATCH_LIST,
            palette=color_palette,
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
            hue="batch_size",
            hue_order=BATCH_LIST,
            palette=color_palette,
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
            hue="batch_size",
            hue_order=BATCH_LIST,
            palette=color_palette,
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
    fig.legend(ax.get_lines(), [str(size) for size in BATCH_LIST], loc="lower center", bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)
    
    colors = sns.color_palette(palette=color_palette, n_colors=len(BATCH_LIST))  # Get a color palette with 3 distinct colors
    patches = [mlines.Line2D([], [], color=colors[i], label=str(batch_size), linewidth=2.2, linestyle='-') for i, batch_size in enumerate(BATCH_LIST)]
    fig.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.03), ncols=len(BATCH_LIST), frameon=False)    

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("new_data_eff/plot_main_time.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib, see http://phyletica.org/matplotlib-fonts/
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    # Create the DataFrame
    results_dir = Path("new_data_eff/")
    #print(results_dir)
    EPISODE_LENGTH = 1000
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]
    
    
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
    grouped = df.groupby(['env', 'algo', 'run', 'batch_size', 'iteration_block'])
    aggregated_per_run = grouped[metrics].mean().reset_index()

    # Calculate the average time across all runs for each iteration block in each env and algo
    average_time = df.groupby(['env', 'algo', 'batch_size', 'iteration_block'])['time'].mean().reset_index()

    # Merge the average time back with the per-run metrics
    final_df = pd.merge(aggregated_per_run, average_time, on=['env', 'algo', 'batch_size', 'iteration_block'])

    # Sorting for better structure and resetting index for cleanliness
    final_df = final_df.sort_values(by=['env', 'algo', 'batch_size', 'run', 'iteration_block']).reset_index(drop=True)

    # Plot
    plot(final_df)