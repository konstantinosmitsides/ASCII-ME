"""
Implementation of the ASCII-ME algorithm for Quality-Diversity optimization.

This script combines reinforcement learning policy gradient methods with
MAP-Elites for quality-diversity optimization, using JAX for efficient computation.
"""

import os
import time
from dataclasses import dataclass
from functools import partial
from math import floor
from typing import Callable
import pickle

import hydra
from omegaconf import OmegaConf
import jax
import jax.numpy as jnp
import numpy as np

from hydra.core.config_store import ConfigStore
from qdax.core.map_elites__ import MAPElites
from qdax.types import RNGKey
from qdax.utils.sampling import sampling 
from qdax.core.containers.mapelites_repertoire_ import compute_cvt_centroids
from qdax.core.neuroevolution.networks.networks import MLPMCPG
from qdax.core.emitters.ascii_me_emitter import ASCIIMEConfig, ASCIIMEEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition 
from qdax.environments_v1 import behavior_descriptor_extractor
from qdax.tasks.brax_envs_advanced_baseline_time_step import reset_based_scoring_function_brax_envs as scoring_function
from utils import Config, get_env
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.utils.metrics import CSVLogger, default_qd_metrics
import matplotlib.pyplot as plt

# Constants
EPS = 1e-8

@hydra.main(version_base="1.2", config_path="configs", config_name="ascii_me")
def main(config: Config) -> None:
    """
    Main function to run the ASCII-ME algorithm.
    
    This function initializes the environment, policy network, and MAP-Elites algorithm,
    then runs the main training loop to build a diverse repertoire of policies.
    
    Args:
        config: Configuration object containing all parameters for the experiment
    """
    
    # --------------------- INITIALIZATION ---------------------
    # Initialize random key for reproducibility
    random_key = jax.random.PRNGKey(config.seed)

    # Initialize environment
    env = get_env(config)
    reset_fn = jax.jit(env.reset)

    # Compute the centroids for MAP-Elites grid
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        random_key=random_key,
    )
    
    # --------------------- POLICY NETWORK ---------------------
    # Initialize policy network based on configuration
    if config.init == "uniform":
        policy_network = MLPMCPG(
            action_dim=env.action_size,
            activation=config.algo.activation,
            no_neurons=config.algo.no_neurons,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_init=jax.nn.initializers.lecun_uniform(),
        )
    else:
        policy_network = MLPMCPG(
            action_dim=env.action_size,
            activation=config.algo.activation,
            no_neurons=config.algo.no_neurons,
            std=config.algo.std,
        )
    
    # Initialize population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch_obs = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    # Log parameter count for information
    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)

    # --------------------- ENVIRONMENT INTERACTION FUNCTIONS ---------------------
    @jax.jit
    def play_step_fn(env_state, policy_params, random_key):
        """
        Function to execute a single step in the environment using the policy.
        
        Args:
            env_state: Current environment state
            policy_params: Parameters of the policy network
            random_key: JAX random key
            
        Returns:
            Tuple containing (next_state, policy_params, random_key) and transition data
        """
        _, action = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, action)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=action,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return (next_state, policy_params, random_key), transition

    def get_n_offspring_added(metrics):
        """
        Count the number of offspring added by different mutation methods.
        
        Args:
            metrics: Dictionary of metrics from the MAP-Elites algorithm
            
        Returns:
            Tuple containing counts of GA and QPG offspring added
        """
        split = jnp.cumsum(jnp.array([emitter.batch_size for emitter in map_elites._emitter.emitters]))
        split = jnp.split(metrics["is_offspring_added"], split, axis=-1)[:-1]
        
        if config.algo.proportion_mutation_ga == 0:
            return (jnp.array([0]), jnp.sum(split[0], axis=-1))
        elif config.algo.proportion_mutation_ga == 1:
            return (jnp.sum(split[0], axis=-1), jnp.array([0]))
        else:
            return (jnp.sum(split[1], axis=-1), jnp.sum(split[0], axis=-1))

    # --------------------- SCORING AND METRICS ---------------------
    # Prepare the scoring function for evaluating policies
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    
    reward_offset = 0

    # Define metrics function for evaluating the quality-diversity performance
    metrics_function = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # --------------------- EMITTER CONFIGURATION ---------------------
    # Configure the ASCII-ME emitter
    ascii_me_config = ASCIIMEConfig(
        proportion_mutation_ga=config.algo.proportion_mutation_ga,
        no_agents=config.batch_size,
        buffer_sample_batch_size=config.algo.buffer_sample_batch_size,
        no_epochs=config.algo.no_epochs,
        learning_rate=config.algo.learning_rate,
        clip_param=config.algo.clip_param,
        discount_rate=config.algo.discount_rate,
        cosine_similarity=config.algo.cosine_similarity,
        std=config.algo.std,
    )
    
    # Define the variation function for mutation
    variation_fn = partial(
        isoline_variation, 
        iso_sigma=config.algo.iso_sigma, 
        line_sigma=config.algo.line_sigma
    )
    
    # Create the ASCII-ME emitter
    ascii_me_emitter = ASCIIMEEmitter(
        config=ascii_me_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
    )
    
    # --------------------- MAP-ELITES SETUP ---------------------
    # Instantiate MAP-Elites algorithm
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=ascii_me_emitter,
        metrics_function=metrics_function,
    )

    # Initialize repertoire with random policies
    repertoire, emitter_state, random_key = map_elites.init(init_params, centroids, random_key)

    # --------------------- LOGGING SETUP ---------------------
    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    # Initialize metrics dictionary
    metrics = dict.fromkeys(
        [
            "iteration", 
            "qd_score", 
            "coverage", 
            "max_fitness",  
            "time", 
            "evaluation", 
            "ga_offspring_added", 
            "qpg_offspring_added"
        ], 
        jnp.array([])
    )
    
    # Setup CSV logger
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )
    
    def plot_metrics_vs_iterations(metrics, log_period):
        """
        Plot metrics over iterations and save figures.
        
        Args:
            metrics: Dictionary of metrics
            log_period: Logging frequency
        """
        iterations = jnp.arange(1, 1 + len(metrics["time"]), dtype=jnp.int32)

        for metric_name, metric_values in metrics.items():
            if metric_name in ["iteration", "evaluation"]:
                continue
            plt.figure()
            plt.plot(iterations, metric_values, label=metric_name)
            plt.xlabel("Iteration")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} vs Iterations")
            plt.legend()
            plt.savefig(f"./Plots/{metric_name}_vs_iterations.png")
            plt.close()
    
    # --------------------- MAIN TRAINING LOOP ---------------------
    map_elites_scan_update = map_elites.scan_update
    eval_num = config.batch_size 
    metrics_file_path = "./metrics_incremental.pickle"

    cumulative_time = 0
    for i in range(num_loops):
        # Time each loop iteration
        start_time = time.time()
        
        # Run MAP-Elites for log_period iterations
        (repertoire, emitter_state, random_key,), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time
        cumulative_time += timelapse

        # Update metrics with iteration information
        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluation"] = jnp.arange(1+log_period*eval_num*i, 1+log_period*eval_num*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(cumulative_time, log_period)
        current_metrics["ga_offspring_added"], current_metrics["qpg_offspring_added"] = get_n_offspring_added(current_metrics)
        
        # Clean up metrics
        del current_metrics["is_offspring_added"]
        current_metrics_cpu = jax.tree_util.tree_map(lambda x: np.array(x), current_metrics)

        # Save incremental metrics to file
        with open(metrics_file_path, "ab") as f:
            pickle.dump(current_metrics_cpu, f)

        # Log metrics for this iteration
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], current_metrics_cpu)
        log_metrics["qpg_offspring_added"] = np.sum(current_metrics["qpg_offspring_added"])
        log_metrics["ga_offspring_added"] = np.sum(current_metrics["ga_offspring_added"])
        csv_logger.log(log_metrics)

    # --------------------- FINAL METRICS PROCESSING ---------------------
    # Combine all incremental metrics into a single structure
    all_metrics = {}
    with open(metrics_file_path, "rb") as f:
        # Read all chunks of metrics
        metrics_list = []
        try:
            while True:
                m = pickle.load(f)
                metrics_list.append(m)
        except EOFError:
            pass

    # Combine all metrics arrays across the loaded chunks
    for key in metrics_list[0].keys():
        all_metrics[key] = np.concatenate([m[key] for m in metrics_list], axis=0)

    # Save combined metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(all_metrics, metrics_file)

    # Save final repertoire
    os.makedirs("./repertoire/", exist_ok=True)
    repertoire.save(path="./repertoire/")

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()