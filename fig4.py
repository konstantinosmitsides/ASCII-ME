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
    "hopper_uni_250",
    "walker2d_uni_250",
    "ant_uni_250",
    "anttrap_omni_250",
    "ant_omni_250",
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

def aggregate_and_smooth(df, window_size=100):
    # Empty DataFrame to store results
    smoothed_data = pd.DataFrame()
    
    # Loop over each unique environment and algorithm combination
    for (env, algo), group in df.groupby(['env', 'algo']):
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
    # Add grid
    ax.grid(which="major", axis="y", color="0.9")
    return ax

def plot(final_df, raw_df):
    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=len(ENV_LIST), sharex=True, squeeze=False, figsize=(25, 10))

    # Create formatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))

    # Define checkpoints: (evaluation_count, marker, label)
    checkpoints = [
        (1_000_000, 'X', '1M'),
        (4_000_000, '^', '4M'),
        (7_000_000, 'o', '7M'),
        (10_000_000, 's', '10M')
    ]
    
    for col, env in enumerate(ENV_LIST):
        # Filter to the environment
        df_raw_env = raw_df[(raw_df["env"] == env) & (raw_df["algo"].isin(ALGO_LIST))]

        # --- COLLECT TIMES AT WHICH EACH ALGO REACHES 1M EVALS ---
        ### CHANGED / ADDED ###
        times_1m = {}
        for algo in ALGO_LIST:
            df_raw_algo = df_raw_env[df_raw_env["algo"] == algo]
            if (df_raw_algo["num_evaluations"] >= 1_000_000).any():
                # Earliest time at which this algo >= 1M evaluations
                df_1m = df_raw_algo[df_raw_algo["num_evaluations"] >= 1_000_000]
                times_1m[algo] = df_1m["time"].min()
            else:
                # If it never reached 1M, ignore or set to None
                times_1m[algo] = None

        # Among the algorithms that *do* reach 1M, find the slowest
        # i.e. the largest time_1m
        valid_times = {k: v for k, v in times_1m.items() if v is not None}
        if len(valid_times) > 0:
            slowest_algo = max(valid_times, key=valid_times.get)  # pick algo with largest time
            max_time_1m = valid_times[slowest_algo]
        else:
            # If no algorithm reaches 1M, use overall max time
            max_time_1m = df_raw_env["time"].max()
            slowest_algo = None  # no slowest in terms of 1M because none reached 1M

        # Create a list of algorithms to plot that excludes the slowest
        ### CHANGED / ADDED ###
        if slowest_algo is not None:
            adjusted_algo_list = [algo for algo in ALGO_LIST if algo != slowest_algo]
        else:
            adjusted_algo_list = ALGO_LIST

        # Filter final_df and raw_df to only show data up to slowest algo's time,
        # but *exclude* the slowest algo from the data.
        df_plot = final_df[
            (final_df["env"] == env)
            & (final_df["time"] <= max_time_1m)
            & (final_df["algo"].isin(adjusted_algo_list))
        ]
        df_raw_env = df_raw_env[
            (df_raw_env["time"] <= max_time_1m)
            & (df_raw_env["algo"].isin(adjusted_algo_list))
        ]

        # Set title
        axes[0, col].set_title(ENV_DICT[env])
        # Set xlabel on the bottom row
        axes[2, col].set_xlabel(XLABEL)
        axes[2, col].xaxis.set_major_formatter(formatter)

        # QD score (top row)
        axes[0, col].yaxis.set_major_formatter(formatter)
        sns.lineplot(
            data=df_plot,
            x="time",
            y="qd_score",
            hue="algo",
            hue_order=adjusted_algo_list,  ### CHANGED / ADDED ###
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[0, col],
        )
        if col == 0:
            axes[0, col].set_ylabel("QD score")
        else:
            axes[0, col].set_ylabel(None)
        customize_axis(axes[0, col])

        # Coverage (middle row)
        axes[1, col].set_ylim(0., 1.05)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        sns.lineplot(
            data=df_plot,
            x="time",
            y="coverage",
            hue="algo",
            hue_order=adjusted_algo_list,  ### CHANGED / ADDED ###
            estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            legend=False,
            ax=axes[1, col],
        )
        if col == 0:
            axes[1, col].set_ylabel("Coverage")
        else:
            axes[1, col].set_ylabel(None)
        customize_axis(axes[1, col])

        # Max fitness (bottom row)
        if col == 0:
            ax = sns.lineplot(
                data=df_plot,
                x="time",
                y="max_fitness",
                hue="algo",
                hue_order=ALGO_LIST,
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                legend=False,
                ax=axes[2, col],
            )
        else:
            sns.lineplot(
                data=df_plot,
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
        customize_axis(axes[2, col])

        # Now retrieve the line handles from the three rows:
        qd_score_handles = axes[0, col].get_lines()
        coverage_handles = axes[1, col].get_lines()
        fitness_handles = axes[2, col].get_lines()

        # Place checkpoint markers
        for algo_idx, algo in enumerate(adjusted_algo_list):  ### CHANGED / ADDED ###
            df_raw_algo = df_raw_env[df_raw_env["algo"] == algo]
            df_plot_algo = df_plot[df_plot["algo"] == algo]

            if len(df_raw_algo) == 0 or len(df_plot_algo) == 0:
                continue

            # Colors in each row
            line_color_qd = qd_score_handles[algo_idx].get_color()
            line_color_cov = coverage_handles[algo_idx].get_color()
            line_color_fit = fitness_handles[algo_idx].get_color()

            for (cp_eval, cp_marker, cp_label) in checkpoints:
                # Check if the algorithm ever reached this evaluation count
                if not (df_raw_algo["num_evaluations"] >= cp_eval).any():
                    continue

                # Find closest time in raw data for this algo at cp_eval
                closest_idx = (df_raw_algo["num_evaluations"] - cp_eval).abs().idxmin()
                cp_time = df_raw_algo.loc[closest_idx, "time"]

                # Only plot checkpoints if within our chosen max_time_1m
                if cp_time <= max_time_1m:
                    # Find the iteration block in final_df that matches cp_time
                    closest_time_idx = (df_plot_algo["time"] - cp_time).abs().idxmin()
                    closest_iteration_block = df_plot_algo.loc[closest_time_idx, "iteration_block"]

                    # --- QD Score (row 0) ---
                    cp_y_qd = df_plot_algo.loc[
                        df_plot_algo["iteration_block"] == closest_iteration_block, 
                        "qd_score"
                    ].median()

                    axes[0, col].plot(
                        cp_time,
                        cp_y_qd,
                        marker=cp_marker,
                        markersize=10,
                        markeredgecolor='white',
                        markeredgewidth=1.5,
                        color=line_color_qd,
                        alpha=0.9,
                        label=cp_label if (col == 0 and algo == adjusted_algo_list[0]) else ""
                    )

                    # --- Coverage (row 1) ---
                    cp_y_cov = df_plot_algo.loc[
                        df_plot_algo["iteration_block"] == closest_iteration_block, 
                        "coverage"
                    ].median()

                    axes[1, col].plot(
                        cp_time,
                        cp_y_cov,
                        marker=cp_marker,
                        markersize=10,
                        markeredgecolor='white',
                        markeredgewidth=1.5,
                        color=line_color_cov,
                        alpha=0.9,
                        label=""
                    )

                    # --- Max fitness (row 2) ---
                    cp_y_fit = df_plot_algo.loc[
                        df_plot_algo["iteration_block"] == closest_iteration_block, 
                        "max_fitness"
                    ].median()

                    axes[2, col].plot(
                        cp_time,
                        cp_y_fit,
                        marker=cp_marker,
                        markersize=10,
                        markeredgecolor='white',
                        markeredgewidth=1.5,
                        color=line_color_fit,
                        alpha=0.9,
                        label=""
                    )

    # Legend for the algorithms
    fig.legend(
        ax.get_lines(), 
        [ALGO_DICT.get(algo, algo) for algo in ALGO_LIST[:-1]],  # still show original names
        loc="lower center", 
        bbox_to_anchor=(0.5, -0.1), 
        ncols=len(ALGO_LIST)-1, 
        frameon=False
    )
    
    # Create a separate legend for the checkpoints
    handles, labels = axes[0,0].get_legend_handles_labels()
    checkpoint_handles = []
    checkpoint_labels = []
    for h, l in zip(handles, labels):
        if l in ['1M', '4M', '7M', '10M']:
            checkpoint_handles.append(h)
            checkpoint_labels.append(l)

    if checkpoint_handles:
        fig.legend(
            checkpoint_handles, 
            checkpoint_labels, 
            loc="lower center", 
            bbox_to_anchor=(0.5, -0.2), 
            ncols=len(checkpoint_labels), 
            frameon=False
        )

    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig("fig4/output/GA_ablation.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    results_dir = Path("fig4/output/")
    EPISODE_LENGTH = 250
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]

    df['iteration_block'] = df['iteration']

    metrics = ['qd_score', 'coverage', 'max_fitness']
    grouped = df.groupby(['env', 'algo', 'run', 'iteration_block'])
    aggregated_per_run = grouped[metrics].mean().reset_index()

    average_time = df.groupby(['env', 'algo', 'iteration_block'])['time'].mean().reset_index()
    final_df = pd.merge(aggregated_per_run, average_time, on=['env', 'algo', 'iteration_block'])
    final_df = final_df.sort_values(by=['env', 'algo', 'run', 'iteration_block']).reset_index(drop=True)

    # Plot with checkpoints
    plot(final_df, df)