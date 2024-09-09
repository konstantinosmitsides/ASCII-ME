import sys
sys.path.append("/project/")

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, ranksums, mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

BATCH_LIST = [
    1024,
    4096,
    16384,
    32768,
]

ALGO_LIST = [
    #"mcpg_me",
    #"mcpg_me_no_normalizer",
    #"mcpg_me_no_baseline",
    #"mcpg_me_no_ppo_loss",
    #"dcg_me",
    #"dcg_me_gecco",
    "pga_me",
    #"qd_pg",
    #"me_es",
    #"memes",
    #"me",
]

METRICS_LIST = [
    "qd_score",
    #"coverage",
    #"max_fitness",
    "iteration",
    "time"
]

P_VALUE_LIST = [
    
    ["qd_score", "walker2d_uni_250", 1024, 4096],
    ["qd_score", "walker2d_uni_250", 1024, 16384],
    ["qd_score", "walker2d_uni_250", 1024, 32768],
    ["qd_score", "walker2d_uni_250", 4096, 16384],
    ["qd_score", "walker2d_uni_250", 4096, 32768],
    ["qd_score", "walker2d_uni_250", 16384, 32768],
    
    ["iteration", "walker2d_uni_250", 1024, 4096],
    ["iteration", "walker2d_uni_250", 1024, 16384],
    ["iteration", "walker2d_uni_250", 1024, 32768],
    ["iteration", "walker2d_uni_250", 4096, 16384],
    ["iteration", "walker2d_uni_250", 4096, 32768],
    ["iteration", "walker2d_uni_250", 16384, 32768],
    
    ["time", "walker2d_uni_250", 1024, 4096],
    ["time", "walker2d_uni_250", 1024, 16384],
    ["time", "walker2d_uni_250", 1024, 32768],
    ["time", "walker2d_uni_250", 4096, 16384],
    ["time", "walker2d_uni_250", 4096, 32768],
    ["time", "walker2d_uni_250", 16384, 32768],
    
    ["qd_score", "ant_uni_250", 1024, 4096],
    ["qd_score", "ant_uni_250", 1024, 16384],
    ["qd_score", "ant_uni_250", 1024, 32768],
    ["qd_score", "ant_uni_250", 4096, 16384],
    ["qd_score", "ant_uni_250", 4096, 32768],
    ["qd_score", "ant_uni_250", 16384, 32768],
    
    ["iteration", "ant_uni_250", 1024, 4096],
    ["iteration", "ant_uni_250", 1024, 16384],
    ["iteration", "ant_uni_250", 1024, 32768],
    ["iteration", "ant_uni_250", 4096, 16384],
    ["iteration", "ant_uni_250", 4096, 32768],
    ["iteration", "ant_uni_250", 16384, 32768],
    
    ["time", "ant_uni_250", 1024, 4096],
    ["time", "ant_uni_250", 1024, 16384],
    ["time", "ant_uni_250", 1024, 32768],
    ["time", "ant_uni_250", 4096, 16384],
    ["time", "ant_uni_250", 4096, 32768],
    ["time", "ant_uni_250", 16384, 32768],
    
    ["qd_score", "ant_omni_250", 1024, 4096],
    ["qd_score", "ant_omni_250", 1024, 16384],
    ["qd_score", "ant_omni_250", 1024, 32768],
    ["qd_score", "ant_omni_250", 4096, 16384],
    ["qd_score", "ant_omni_250", 4096, 32768],
    ["qd_score", "ant_omni_250", 16384, 32768],
    
    ["iteration", "ant_omni_250", 1024, 4096],
    ["iteration", "ant_omni_250", 1024, 16384],
    ["iteration", "ant_omni_250", 1024, 32768],
    ["iteration", "ant_omni_250", 4096, 16384],
    ["iteration", "ant_omni_250", 4096, 32768],
    ["iteration", "ant_omni_250", 16384, 32768],
    
    ["time", "ant_omni_250", 1024, 4096],
    ["time", "ant_omni_250", 1024, 16384],
    ["time", "ant_omni_250", 1024, 32768],
    ["time", "ant_omni_250", 4096, 16384],
    ["time", "ant_omni_250", 4096, 32768],
    ["time", "ant_omni_250", 16384, 32768],
    
    #['qd_score', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['qd_score', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['qd_score', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['coverage', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['coverage', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['coverage', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['max_fitness', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['max_fitness', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['max_fitness', 'walker2d_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['qd_score', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['qd_score', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['qd_score', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['coverage', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['coverage', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['coverage', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['max_fitness', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['max_fitness', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['max_fitness', 'ant_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['qd_score', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['qd_score', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['qd_score', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['coverage', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['coverage', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['coverage', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],
    
    #['max_fitness', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_normalizer'],
    #['max_fitness', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_baseline'],
    #['max_fitness', 'hopper_uni_1000', 'mcpg_me', 'mcpg_me_no_ppo_loss'],

    
    #["time", "ant_omni_250", "mcpg_me", "dcg_me"],
    #["time", "ant_omni_250", "mcpg_me", "pga_me"],
    #["time", "ant_omni_250", "mcpg_me", "me"],
    #["time", "ant_omni_250", "mcpg_me", "memes"],


    #["time", "anttrap_omni_250", "mcpg_me", "pga_me"],
    #["time", "anttrap_omni_250", "mcpg_me", "dcg_me"],
    #["time", "anttrap_omni_250", "mcpg_me", "me"],
    #["time", "anttrap_omni_250", "mcpg_me", "memes"],


    #["time", "ant_uni_250", "mcpg_me", "pga_me"],
    #["time", "ant_uni_250", "mcpg_me", "dcg_me"],
    #["time", "ant_uni_250", "mcpg_me", "me"],
    #["time", "ant_uni_250", "mcpg_me", "memes"],

    #["time", "walker2d_uni_250", "mcpg_me", "pga_me"],
    #["time", "walker2d_uni_250", "mcpg_me", "dcg_me"],
    #["time", "walker2d_uni_250", "mcpg_me", "me"],
    #["time", "walker2d_uni_250", "mcpg_me", "memes"],


    #["time", "hopper_uni_250", "mcpg_me", "pga_me"],
    #["time", "hopper_uni_250", "mcpg_me", "dcg_me"],
    #["time", "hopper_uni_250", "mcpg_me", "me"],
    #["time", "hopper_uni_250", "mcpg_me", "memes"],
    

    #["time", "ant_uni_1000", "mcpg_me", "pga_me"],
    #["time", "ant_uni_1000", "mcpg_me", "dcg_me"],
    #["time", "ant_uni_1000", "mcpg_me", "me"],
    #["time", "ant_uni_1000", "mcpg_me", "memes"],

    #["time", "walker2d_uni_1000", "mcpg_me", "pga_me"],
    #["time", "walker2d_uni_1000", "mcpg_me", "dcg_me"],
    #["time", "walker2d_uni_1000", "mcpg_me", "me"],
    #["time", "walker2d_uni_1000", "mcpg_me", "memes"],


    #["time", "hopper_uni_1000", "mcpg_me", "pga_me"],
    #["time", "hopper_uni_1000", "mcpg_me", "dcg_me"],
    #["time", "hopper_uni_1000", "mcpg_me", "me"],
    #["time", "hopper_uni_1000", "mcpg_me", "memes"],
    
    #["qd_score", "ant_omni_250", "mcpg_me", "dcg_me"],
    #["qd_score", "ant_omni_250", "mcpg_me", "pga_me"],
    #["qd_score", "ant_omni_250", "mcpg_me", "me"],
    #["qd_score", "ant_omni_250", "mcpg_me", "memes"],


    #["qd_score", "anttrap_omni_250", "mcpg_me", "dcg_me"],
    #["qd_score", "anttrap_omni_250", "mcpg_me", "pga_me"],
    #["qd_score", "anttrap_omni_250", "mcpg_me", "memes"],
    #["qd_score", "anttrap_omni_250", "mcpg_me", "me"],


    #["qd_score", "ant_uni_250", "mcpg_me", "pga_me"],
    #["qd_score", "ant_uni_250", "mcpg_me", "dcg_me"],
    #["qd_score", "ant_uni_250", "mcpg_me", "me"],
    #["qd_score", "ant_uni_250", "mcpg_me", "memes"],

    #["qd_score", "walker2d_uni_250", "mcpg_me", "pga_me"],
    #["qd_score", "walker2d_uni_250", "mcpg_me", "dcg_me"],
    #["qd_score", "walker2d_uni_250", "mcpg_me", "me"],
    #["qd_score", "walker2d_uni_250", "mcpg_me", "memes"],


    #["qd_score", "hopper_uni_250", "mcpg_me", "pga_me"],
    #["qd_score", "hopper_uni_250", "mcpg_me", "dcg_me"],
    #["qd_score", "hopper_uni_250", "mcpg_me", "me"],
    #["qd_score", "hopper_uni_250", "mcpg_me", "memes"],


    #["qd_score", "ant_uni_1000", "mcpg_me", "pga_me"],
    #["qd_score", "ant_uni_1000", "mcpg_me", "dcg_me"],
    #["qd_score", "ant_uni_1000", "mcpg_me", "me"],
    #["qd_score", "ant_uni_1000", "mcpg_me", "memes"],


    #["qd_score", "walker2d_uni_1000", "mcpg_me", "dcg_me"],
    #["qd_score", "walker2d_uni_1000", "mcpg_me", "pga_me"],
    #["qd_score", "walker2d_uni_1000", "mcpg_me", "me"],
    #["qd_score", "walker2d_uni_1000", "mcpg_me", "memes"],


    #["qd_score", "hopper_uni_1000", "mcpg_me", "dcg_me"],
    #["qd_score", "hopper_uni_1000", "mcpg_me", "pga_me"],
    #["qd_score", "hopper_uni_1000", "mcpg_me", "me"],
    #["qd_score", "hopper_uni_1000", "mcpg_me", "memes"],
    
    #["coverage", "ant_omni_250", "mcpg_me", "pga_me"],
    #["coverage", "ant_omni_250", "mcpg_me", "dcg_me"],
    #["coverage", "ant_omni_250", "mcpg_me", "me"],
    #["coverage", "ant_omni_250", "mcpg_me", "memes"],
    
    #["coverage", "anttrap_omni_250", "mcpg_me", "pga_me"],
    #["coverage", "anttrap_omni_250", "mcpg_me", "dcg_me"],
    #["coverage", "anttrap_omni_250", "mcpg_me", "me"],
    #["coverage", "anttrap_omni_250", "mcpg_me", "memes"],
    
    #["coverage", "ant_uni_250", "mcpg_me", "pga_me"],
    #["coverage", "ant_uni_250", "mcpg_me", "dcg_me"],
    #["coverage", "ant_uni_250", "mcpg_me", "me"],
    #["coverage", "ant_uni_250", "mcpg_me", "memes"],
    
    #["coverage", "walker2d_uni_250", "mcpg_me", "pga_me"],
    #["coverage", "walker2d_uni_250", "mcpg_me", "dcg_me"],
    #["coverage", "walker2d_uni_250", "mcpg_me", "me"],
    #["coverage", "walker2d_uni_250", "mcpg_me", "memes"],
    
    #["coverage", "hopper_uni_250", "mcpg_me", "pga_me"],
    #["coverage", "hopper_uni_250", "mcpg_me", "dcg_me"],
    #["coverage", "hopper_uni_250", "mcpg_me", "me"],
    #["coverage", "hopper_uni_250", "mcpg_me", "memes"],
    
    #["coverage", "ant_uni_1000", "mcpg_me", "pga_me"],
    #["coverage", "ant_uni_1000", "mcpg_me", "dcg_me"],
    #["coverage", "ant_uni_1000", "mcpg_me", "me"],
    #["coverage", "ant_uni_1000", "mcpg_me", "memes"],
    
    #["coverage", "walker2d_uni_1000", "mcpg_me", "pga_me"],
    #["coverage", "walker2d_uni_1000", "mcpg_me", "dcg_me"],
    #["coverage", "walker2d_uni_1000", "mcpg_me", "me"],
    #["coverage", "walker2d_uni_1000", "mcpg_me", "memes"],
    
    #["coverage", "hopper_uni_1000", "mcpg_me", "pga_me"],
    #["coverage", "hopper_uni_1000", "mcpg_me", "dcg_me"],
    #["coverage", "hopper_uni_1000", "mcpg_me", "me"],
    #["coverage", "hopper_uni_1000", "mcpg_me", "memes"],
    
    #["max_fitness", "ant_omni_250", "mcpg_me", "pga_me"],
    #["max_fitness", "ant_omni_250", "mcpg_me", "dcg_me"],
    #["max_fitness", "ant_omni_250", "mcpg_me", "me"],
    #["max_fitness", "ant_omni_250", "mcpg_me", "memes"],
    
    #["max_fitness", "anttrap_omni_250", "mcpg_me", "dcg_me"],
    #["max_fitness", "anttrap_omni_250", "mcpg_me", "pga_me"],
    #["max_fitness", "anttrap_omni_250", "mcpg_me", "memes"],
    #["max_fitness", "anttrap_omni_250", "mcpg_me", "me"],
    
    #["max_fitness", "ant_uni_250", "mcpg_me", "dcg_me"],
    #["max_fitness", "ant_uni_250", "mcpg_me", "pga_me"],
    #["max_fitness", "ant_uni_250", "mcpg_me", "me"],
    #["max_fitness", "ant_uni_250", "mcpg_me", "memes"],
    
    #["max_fitness", "walker2d_uni_250", "mcpg_me", "pga_me"],
    #["max_fitness", "walker2d_uni_250", "mcpg_me", "dcg_me"],
    #["max_fitness", "walker2d_uni_250", "mcpg_me", "me"],
    #["max_fitness", "walker2d_uni_250", "mcpg_me", "memes"],
    
    #["max_fitness", "hopper_uni_250", "mcpg_me", "pga_me"],
    #["max_fitness", "hopper_uni_250", "mcpg_me", "dcg_me"],
    #["max_fitness", "hopper_uni_250", "mcpg_me", "me"],
    #["max_fitness", "hopper_uni_250", "mcpg_me", "memes"],
    
    #["max_fitness", "ant_uni_1000", "mcpg_me", "pga_me"],
    #["max_fitness", "ant_uni_1000", "mcpg_me", "me"],
    #["max_fitness", "ant_uni_1000", "mcpg_me", "dcg_me"],
    #["max_fitness", "ant_uni_1000", "mcpg_me", "memes"],
    
    #["max_fitness", "walker2d_uni_1000", "mcpg_me", "pga_me"],
    #["max_fitness", "walker2d_uni_1000", "mcpg_me", "dcg_me"],
    #["max_fitness", "walker2d_uni_1000", "mcpg_me", "me"],
    #["max_fitness", "walker2d_uni_1000", "mcpg_me", "memes"],
    
    #["max_fitness", "hopper_uni_1000", "mcpg_me", "pga_me"],
    #["max_fitness", "hopper_uni_1000", "mcpg_me", "dcg_me"],
    #["max_fitness", "hopper_uni_1000", "mcpg_me", "me"],
    #["max_fitness", "hopper_uni_1000", "mcpg_me", "memes"],
]

#P_VALUE_LIST = [[*item[:2], int(item[2]), int(item[3])] for item in P_VALUE_LIST]


if __name__ == "__main__":
    # Create the DataFrame
    results_dir = Path("scalability/output/")
    
    EPISODE_LENGTH = 250

    df = get_df(results_dir, EPISODE_LENGTH)

    # Filter
    df = df[df["algo"].isin(ALGO_LIST)]
    df = df[df["num_evaluations"] <= 1_001_400]

    # Keep only the last iteration
    idx = df.groupby(["env", "algo", "run"])["iteration"].idxmax()
    df = df.loc[idx]

    # Compute p-values
    algo = "pga_me"
    p_value_df = pd.DataFrame(columns=["metric", "env", "batch_size_1", "batch_size_2", "p_value"])
    for metric in METRICS_LIST:
        for env in ENV_LIST:
            for batch_size_1 in BATCH_LIST:
                for batch_size_2 in BATCH_LIST:
                    stat = mannwhitneyu(
                        df[(df["env"] == env) & (df["algo"] == algo) & (df["batch_size"] == batch_size_1)][metric],
                        df[(df["env"] == env) & (df["algo"] == algo) & (df["batch_size"] == batch_size_2)][metric],
                   )
                    p_value_df.loc[len(p_value_df)] = {"metric": metric, "env": env, "batch_size_1": batch_size_1, "batch_size_2": batch_size_2, "p_value": stat.pvalue}
            #for algo_1 in ALGO_LIST:
            #    for algo_2 in ALGO_LIST:
            #        stat = mannwhitneyu(
            #            df[(df["env"] == env) & (df["algo"] == algo_1)][metric],
            #            df[(df["env"] == env) & (df["algo"] == algo_2)][metric],
            #        )
            #        p_value_df.loc[len(p_value_df)] = {"metric": metric, "env": env, "algo_1": algo_1, "algo_2": algo_2, "p_value": stat.pvalue}

    # Filter p-values
    p_value_df.set_index(["metric", "env", "batch_size_1", "batch_size_2"], inplace=True)
    p_value_df = p_value_df.loc[P_VALUE_LIST]

    # Correct p-values
    p_value_df.reset_index(inplace=True)
    p_value_df["p_value_corrected"] = multipletests(p_value_df["p_value"], method="holm")[1]
    p_value_df = p_value_df.pivot(index=["env", "batch_size_1", "batch_size_2"], columns="metric", values="p_value_corrected")

    # Save p-values
    p_value_df.to_csv("scalability/output/p_value_pga_me.csv")