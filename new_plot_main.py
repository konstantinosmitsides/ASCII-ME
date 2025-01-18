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
    # "walker2d_uni_250",
    # "ant_uni_250",
    # "hopper_uni_250",
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
    "dcg_me",
    "pga_me",
    "me",
    "memes",
    'mcpg_only',
    "ppga",
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
    "ppga" : "PPGA",
    "mcpg_only": "MCPG-Only"
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
        if df_row["proportion_mutation_ga"] == 0:
            return "mcpg_only"
            
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

        # Find the maximum time at which any algorithm reaches at least 1M evaluations
        times_1m = []
        for algo in ALGO_LIST:
            df_raw_algo = df_raw_env[df_raw_env["algo"] == algo]
            # Find if this algo reaches 1M evaluations
            if (df_raw_algo["num_evaluations"] >= 1_000_000).any():
                # Get the earliest time at which this algo >= 1M evaluations
                df_1m = df_raw_algo[df_raw_algo["num_evaluations"] >= 1_000_000]
                time_1m = df_1m["time"].min()
                times_1m.append(time_1m)
        
        # If no algorithm reaches 1M, we'll just use the max time available
        if len(times_1m) > 0:
            max_time_1m = max(times_1m)
        else:
            max_time_1m = df_raw_env["time"].max()

        # Filter final_df and raw_df to only show data up to max_time_1m
        df_plot = final_df[(final_df["env"] == env) & (final_df["time"] <= max_time_1m)]
        df_raw_env = df_raw_env[df_raw_env["time"] <= max_time_1m]

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
        customize_axis(axes[0, col])

        # Coverage (middle row)
        axes[1, col].set_ylim(0., 1.05)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        sns.lineplot(
            data=df_plot,
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

        # # Retrieve line handles for QD score lines to get colors
        # line_handles = axes[0, col].get_lines()

        # # Place checkpoints on QD score lines
        # for algo_idx, algo in enumerate(ALGO_LIST):
        #     df_raw_algo = df_raw_env[df_raw_env["algo"] == algo]
        #     df_plot_algo = df_plot[df_plot["algo"] == algo]

        #     if len(df_raw_algo) == 0 or len(df_plot_algo) == 0:
        #         continue

        #     line_color = line_handles[algo_idx].get_color()

        #     for (cp_eval, cp_marker, cp_label) in checkpoints:
        #         # Check if the algorithm ever reached this evaluation count
        #         if not (df_raw_algo["num_evaluations"] >= cp_eval).any():
        #             # This algorithm never reached cp_eval, skip plotting this symbol
        #             continue

        #         # Find closest time in raw data for this algo at cp_eval
        #         closest_idx = (df_raw_algo["num_evaluations"] - cp_eval).abs().idxmin()
        #         cp_time = df_raw_algo.loc[closest_idx, "time"]

        #         # Only plot checkpoints if within the max_time_1m
        #         if cp_time <= max_time_1m:
        #             # Find the closest iteration block to cp_time
        #             closest_time_idx = (df_plot_algo["time"] - cp_time).abs().idxmin()
        #             closest_iteration_block = df_plot_algo.loc[closest_time_idx, "iteration_block"]

        #             # Get median qd_score across runs for that iteration_block
        #             cp_y = df_plot_algo.loc[df_plot_algo["iteration_block"] == closest_iteration_block, "qd_score"].median()

        #             # Label only once (first algo and first column)
        #             label_to_use = cp_label if (col == 0 and algo == ALGO_LIST[0]) else ""

        #             # Plot the checkpoint marker on the line
        #             axes[0, col].plot(
        #                 cp_time, cp_y, 
        #                 marker=cp_marker, 
        #                 markersize=10,  # Slightly bigger
        #                 markeredgecolor='white', 
        #                 markeredgewidth=1.5,
        #                 color=line_color, 
        #                 alpha=0.9, 
        #                 label=label_to_use
        #             )

        # After plotting, retrieve line handles from each row
        qd_score_handles = axes[0, col].get_lines()
        coverage_handles = axes[1, col].get_lines()
        fitness_handles = axes[2, col].get_lines()

        # print("Number of plotted lines:", len(qd_score_handles))
        # print("Number of algorithms expected:", len(ALGO_LIST))

        # Loop over each algorithm
        for algo_idx, algo in enumerate(ALGO_LIST):
            df_raw_algo = df_raw_env[df_raw_env["algo"] == algo]
            df_plot_algo = df_plot[df_plot["algo"] == algo]

            if len(df_raw_algo) == 0 or len(df_plot_algo) == 0:
                continue

            # Get the line color for each metric by matching algo_idx
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

                # Only plot checkpoints if within the max_time_1m
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
                        label=cp_label if (col == 0 and algo == ALGO_LIST[0]) else ""
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
                        # e.g. no label here, or same logic as above if desired
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
    fig.legend(ax.get_lines(), [ALGO_DICT.get(algo, algo) for algo in ALGO_LIST],
               loc="lower center", bbox_to_anchor=(0.5, -0.1), ncols=len(ALGO_LIST), frameon=False)
    
    # Create a separate legend for the checkpoints
    handles, labels = axes[0,0].get_legend_handles_labels()
    checkpoint_handles = []
    checkpoint_labels = []
    for h, l in zip(handles, labels):
        if l in ['1M', '4M', '7M', '10M']:
            checkpoint_handles.append(h)
            checkpoint_labels.append(l)

    # Add the checkpoint legend below the main legend (only if checkpoints are present)
    if checkpoint_handles:
        fig.legend(checkpoint_handles, checkpoint_labels, loc="lower center", 
                   bbox_to_anchor=(0.5, -0.2), ncols=len(checkpoint_labels), frameon=False)

    # Aesthetic
    fig.align_ylabels(axes)
    fig.tight_layout()

    # Save plot
    fig.savefig("fig1/output/plot_main_time_ppga.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # Avoid type3 fonts in matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rc("font", size=16)

    results_dir = Path("fig1/output/")
    EPISODE_LENGTH = 250
    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df['algo'] = df.apply(filter, axis=1)
    df = df[df["algo"].isin(ALGO_LIST)]
    #df = df[df["num_evaluations"] <= 1_001_400]

    df['iteration_block'] = df['iteration']
    #df['iteration_block'] = (df['iteration'] - 1) // 10

    metrics = ['qd_score', 'coverage', 'max_fitness']
    grouped = df.groupby(['env', 'algo', 'run', 'iteration_block'])
    aggregated_per_run = grouped[metrics].mean().reset_index()

    average_time = df.groupby(['env', 'algo', 'iteration_block'])['time'].mean().reset_index()
    final_df = pd.merge(aggregated_per_run, average_time, on=['env', 'algo', 'iteration_block'])
    final_df = final_df.sort_values(by=['env', 'algo', 'run', 'iteration_block']).reset_index(drop=True)

    # Plot with checkpoints
    plot(final_df, df)