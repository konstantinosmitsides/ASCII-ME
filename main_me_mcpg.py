import os

os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['WANDB_CACHE_DIR'] = '/tmp/wandb_cache'
os.environ['JAX_LOG_COMPILATION'] = '1'

import logging
import time
from dataclasses import dataclass
from functools import partial
from math import floor
from typing import Any, Dict, Tuple, List, Callable
import pickle
from flax import serialization
#logging.basicConfig(level=logging.DEBUG)
import hydra
from omegaconf import OmegaConf, DictConfig
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.map_elites import MAPElites
from qdax.types import RNGKey, Genotype
from qdax.utils.sampling import sampling 
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.neuroevolution.networks.networks import MLPMCPG
from qdax.core.emitters.me_mcpg_emitter import MEMCPGConfig, MEMCPGEmitter
#from qdax.core.emitters.rein_emitter_advanced import REINaiveConfig, REINaiveEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition, QDMCTransition
from qdax.environments import behavior_descriptor_extractor
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from utils import Config, get_env
from qdax.core.emitters.mutation_operators import isoline_variation
import wandb
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results, plot_2d_map_elites_repertoire
import matplotlib.pyplot as plt
from set_up_brax import get_reward_offset_brax




    


@hydra.main(version_base="1.2", config_path="configs", config_name="me_mcpg")
def main(config: Config) -> None:
    wandb.init(
        project="me-mcpg",
        name=config.alg_name,
        config=OmegaConf.to_container(config, resolve=True),
    )
    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)
    reset_fn = jax.jit(env.reset)

    # Compute the centroids
    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=config.env.min_bd,
        maxval=config.env.max_bd,
        random_key=random_key,
    )
    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes #+ (env.action_size,)
    print(policy_layer_sizes)
    

    
    '''
    policy_network = MLPMCPG(
        hidden_layers_size=policy_layer_sizes,
        action_size=env.action_size,
        activation=jax.nn.tanh,
        hidden_init=jax.nn.initializers.orthogonal(scale=jnp.sqrt(2)),
        mean_init=jax.nn.initializers.orthogonal(scale=0.01),
    )
    '''
    

        
    '''
    policy_network = MLPMCPG(
        hidden_layers_size=policy_layer_sizes,
        action_size=env.action_size,
        activation=jax.nn.tanh,
        hidden_init=jax.nn.initializers.variance_scaling(scale=jnp.sqrt(2), mode='fan_in', distribution='uniform'),
        mean_init=jax.nn.initializers.variance_scaling(scale=0.02*jnp.sqrt(2), mode='fan_in', distribution='uniform'),
    )
    '''

    
    
    policy_network = MLPMCPG(
        hidden_layers_size=policy_layer_sizes,
        action_size=env.action_size,
        activation=jax.nn.tanh,
        hidden_init=jax.nn.initializers.lecun_uniform(),
        mean_init=jax.nn.initializers.lecun_uniform(),
    )
    

    
    
    # Init population of controllers
    
    # maybe consider adding two random keys for each policy
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.no_agents)
    #split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
    #keys1, keys2 = split_keys[:, 0], split_keys[:, 1]
    fake_batch_obs = jnp.zeros(shape=(config.no_agents, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)

    # Define the fonction to play a step with the policy in the environment
    @jax.jit
    def play_step_fn(env_state, policy_params, random_key):
        #random_key, subkey = jax.random.split(random_key)
        actions, logp = policy_network.apply(policy_params, env_state.obs)
        #logp = policy_network.apply(policy_params, env_state.obs, actions, method=policy_network.logp)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDMCTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            logp=logp,
            #desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            #desc_prime=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition




    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    #reward_offset = get_reward_offset_brax(env, config.env_name)
    #print(f"Reward offset: {reward_offset}")
    
    me_scoring_fn = partial(
    sampling,
    scoring_fn=scoring_fn,
    num_samples=config.num_samples,
)

    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )

        # Compute repertoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)
        #qd_score += reward_offset * config.env.episode_length * jnp.sum(1.0 - repertoire_empty)

        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, dem
    
    reward_offset = 0
    
    def recreate_repertoire(
        repertoire: MapElitesRepertoire,
        centroids: jnp.ndarray,
        metrics_fn: Callable,
        random_key: RNGKey,
    ) -> MapElitesRepertoire:
        
        (
            old_qd_score,
            old_max_fitness,
            old_coverage
        ) = metrics_fn(repertoire).values()
        fitnesses, descriptors, extra_scores, random_key = me_scoring_fn(
            repertoire.genotypes, random_key
        )
        new_repertoire = MapElitesRepertoire.init(
            genotypes=repertoire.genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )
        
        (
            new_qd_score,
            new_max_fitness,
            new_coverage,
        ) = metrics_fn(new_repertoire).values()
        
        
        def calculate_percentage_difference(old, new):
            return (abs(new - old) / ((new + old) / 2)) * 100

        qd_score_difference = calculate_percentage_difference(old_qd_score, new_qd_score)
        max_fitness_difference = calculate_percentage_difference(old_max_fitness, new_max_fitness)
        coverage_difference = calculate_percentage_difference(old_coverage, new_coverage)
        
        # Save scores and percentage differences to a file
        with open("./recreated_scores.txt", "w") as file:
            file.write(f"Old QD Score: {old_qd_score}\n")
            file.write(f"New QD Score: {new_qd_score}\n")
            file.write(f"QD Score Percentage Difference: {qd_score_difference}%\n")
            file.write(f"Old Max Fitness: {old_max_fitness}\n")
            file.write(f"New Max Fitness: {new_max_fitness}\n")
            file.write(f"Max Fitness Percentage Difference: {max_fitness_difference}%\n")
            file.write(f"Old Coverage: {old_coverage}\n")
            file.write(f"New Coverage: {new_coverage}\n")
            file.write(f"Coverage Percentage Difference: {coverage_difference}%\n")
            
        if env.behavior_descriptor_length == 2:
            
            fig, _ = plot_2d_map_elites_repertoire(
                centroids=new_repertoire.centroids,
                repertoire_fitnesses=new_repertoire.fitnesses,
                minval=config.env.min_bd,
                maxval=config.env.max_bd,
                repertoire_descriptors=new_repertoire.descriptors,
            )
            
            fig.savefig("./recreated_repertoire_plot.png")
        
        
        
        
        
        
    
        

   
    '''
    def get_n_offspring_added(metrics):
        split = jnp.cumsum(jnp.array([emitter.batch_size for emitter in map_elites._emitter.emitters]))
        split = jnp.split(metrics["is_offspring_added"], split, axis=-1)[:-1]
        qpg_offspring_added, ai_offspring_added = jnp.split(split[0], (split[0].shape[1]-1,), axis=-1)
        return (jnp.sum(split[1], axis=-1), jnp.sum(qpg_offspring_added, axis=-1), jnp.sum(ai_offspring_added, axis=-1))
    '''
    # Get minimum reward value to make sure qd_score are positive
    

    # Define a metrics function
    metrics_function = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # Define the PG-emitter config
    
    me_mcpg_config = MEMCPGConfig(
        proportion_mutation_ga=config.proportion_mutation_ga,
        no_agents=config.no_agents,
        buffer_sample_batch_size=config.buffer_sample_batch_size,
        buffer_add_batch_size=config.buffer_add_batch_size,
        #batch_size=config.batch_size,
        #mini_batch_size=config.mini_batch_size,
        no_epochs=config.no_epochs,
        buffer_size=config.buffer_size,
        learning_rate=config.learning_rate,
        adam_optimizer=config.adam_optimizer,
        clip_param=config.clip_param,
    )
    
    variation_fn = partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )
    
    me_mcpg_emitter = MEMCPGEmitter(
        config=me_mcpg_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
        )
    
    '''
    rein_emitter = REINaiveEmitter(
        config=rein_emitter_config,
        policy_network=policy_network,
        env=env,
        )
    '''
    '''
    me_scoring_fn = partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=config.num_samples,
    )
    '''

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=me_mcpg_emitter,
        metrics_function=metrics_function,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(init_params, centroids, random_key)

    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "qd_score_repertoire", "dem_repertoire", "time", "evaluation"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )
    def plot_metrics_vs_iterations(metrics, log_period):
        iterations = jnp.arange(1, 1 + log_period * len(metrics["time"]), dtype=jnp.int32)

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
    # Main loop
    map_elites_scan_update = map_elites.scan_update
    eval_num = config.no_agents 
    print(f"Number of evaluations per iteration: {eval_num}")
    for i in range(num_loops):
        print(f"Loop {i+1}/{num_loops}")
        start_time = time.time()
        
        (repertoire, emitter_state, random_key,), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        random_key, qd_score_repertoire, dem_repertoire = evaluate_repertoire(random_key, repertoire)

        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluation"] = jnp.arange(1+log_period*eval_num*i, 1+log_period*eval_num*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        current_metrics["qd_score_repertoire"] = jnp.repeat(qd_score_repertoire, log_period)
        current_metrics["dem_repertoire"] = jnp.repeat(dem_repertoire, log_period)
        #current_metrics["ga_offspring_added"], current_metrics["qpg_offspring_added"], current_metrics["ai_offspring_added"] = get_n_offspring_added(current_metrics)
        #del current_metrics["is_offspring_added"]
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        #log_metrics["qpg_offspring_added"] = jnp.sum(current_metrics["qpg_offspring_added"])
        #log_metrics["ga_offspring_added"] = jnp.sum(current_metrics["ga_offspring_added"])
        #log_metrics["ai_offspring_added"] = jnp.sum(current_metrics["ai_offspring_added"])
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)

    # Metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)

    # Repertoire
    os.mkdir("./repertoire/")
    os.mkdir("./Plots/")
    repertoire.save(path="./repertoire/")

    plot_metrics_vs_iterations(metrics, log_period)


    # Plot
    if env.behavior_descriptor_length == 2:
        env_steps = jnp.arange(config.num_iterations) * config.env.episode_length * config.no_agents
        fig, _ = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_bd=config.env.min_bd, max_bd=config.env.max_bd)
        fig.savefig("./Plots/repertoire_plot.png")
        
    recreate_repertoire(repertoire, centroids, metrics_function, random_key)
        
    


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()