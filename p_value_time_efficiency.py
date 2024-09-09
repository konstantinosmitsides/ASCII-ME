import sys
sys.path.append("/project/")

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

BATCH_LIST = [
    1024,
    4096,
    16384,
    32768,
]

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
    #"memes",
    #"me",
]

METRICS_LIST = [
    #"qd_score",
    #"coverage",
    #"max_fitness",
    #"iteration",
    "time"
]

P_VALUE_LIST = [
    

    
    #["time", "ant_omni_250", "mcpg_me", "dcg_me"],
    #["time", "anttrap_omni_250", "mcpg_me", "dcg_me"],
    #["time", "ant_uni_250", "mcpg_me", "pga_me"],
    #["time", "ant_uni_250", "mcpg_me", "dcg_me"],
    ["time", "ant_uni_1000", "mcpg_me", "pga_me"],
    ["time", "ant_uni_1000", "mcpg_me", "dcg_me"],
    ["time", "walker2d_uni_1000", "mcpg_me", "dcg_me"],
]


if __name__ == "__main__":
    # Create the DataFrame
    results_dir = Path("data_time_efficiency/output/")
    
    EPISODE_LENGTH = 1000

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]

    # Keep only the last iteration
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    #min_qd_score = df_last_iteration.groupby(["env", "algo"])["qd_score"].min()
    min_qd_score_mcpg_me = df_last_iteration[df_last_iteration["algo"] == "mcpg_me"]["qd_score"].min()
    
    df = df[df["qd_score"] <= min_qd_score_mcpg_me]
    
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df_last_iteration = df.loc[idx]
    # Extract only the relevant columns for easier readability
    summary_df = df_last_iteration[['env', 'algo', 'time', 'qd_score']]
    df = summary_df

    # Compute p-values
    p_value_df = pd.DataFrame(columns=["metric", "env", "algo_1", "algo_2", "p_value"])
    for metric in METRICS_LIST:
        for env in ENV_LIST:
            for algo_1 in ALGO_LIST:
                for algo_2 in ALGO_LIST:
                    arg_1 = df[(df["env"] == env) & (df["algo"] == algo_1)][metric]
                    arg_2 = df[(df["env"] == env) & (df["algo"] == algo_2)][metric]
                    if arg_1.size > 0 and arg_2.size > 0:
                        stat = mannwhitneyu(
                            arg_1,
                            arg_2,
                        )
                        p_value_df.loc[len(p_value_df)] = {"metric": metric, "env": env, "algo_1": algo_1, "algo_2": algo_2, "p_value": stat.pvalue}

    # Filter p-values
    p_value_df.set_index(["metric", "env", "algo_1", "algo_2"], inplace=True)
    p_value_df = p_value_df.loc[P_VALUE_LIST]

    # Correct p-values
    p_value_df.reset_index(inplace=True)
    p_value_df["p_value_corrected"] = multipletests(p_value_df["p_value"], method="holm")[1]
    p_value_df = p_value_df.pivot(index=["env", "algo_1", "algo_2"], columns="metric", values="p_value_corrected")

    # Save p-values
    p_value_df.to_csv(f"data_time_efficiency/output/p_value_time_efficiency_{EPISODE_LENGTH}.csv")