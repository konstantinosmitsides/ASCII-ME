import os


from typing import Tuple
from dataclasses import dataclass
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.environments import behavior_descriptor_extractor
from qdax.core.map_elites_pga import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results, plot_2d_map_elites_repertoire
from qdax.utils.sampling import sampling 
from qdax.types import RNGKey, Genotype
from typing import Any, Dict, Tuple, List, Callable
import matplotlib.pyplot as plt






import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils import Config, get_env
from jax.lib import xla_bridge


@hydra.main(version_base="1.2", config_path="configs/", config_name="pga_me")
def main(config: Config) -> None:
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
 
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch_obs = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)

    # Define the fonction to play a step with the policy in the environment
    @jax.jit
    def play_step_fn(env_state, policy_params, random_key):
        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            truncations=next_state.info["truncation"],
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return (next_state, policy_params, random_key), transition

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    
    me_scoring_fn = functools.partial(
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

        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, dem

    reward_offset = 0
    @jax.jit
    def evaluate_actor(random_key, actor_params):
        actors_params = jax.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), config.batch_size, axis=0), actor_params)
        fitnesses, _, _, random_key = scoring_fn(actors_params, random_key)
        return random_key, fitnesses.mean()

    def get_n_offspring_added(metrics):
        split = jnp.cumsum(jnp.array([emitter.batch_size for emitter in map_elites._emitter.emitters]))
        split = jnp.split(metrics["is_offspring_added"], split, axis=-1)[:-1]
        qpg_offspring_added, ai_offspring_added = jnp.split(split[0], (split[0].shape[1]-1,), axis=-1)
        return (jnp.sum(split[1], axis=-1), jnp.sum(qpg_offspring_added, axis=-1), jnp.sum(ai_offspring_added, axis=-1))


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

    # Define a metrics function
    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # Define the PG-emitter config
    pga_emitter_config = PGAMEConfig(
        env_batch_size=config.batch_size,
        proportion_mutation_ga=config.algo.proportion_mutation_ga,
        critic_hidden_layer_size=config.algo.critic_hidden_layer_size,
        num_critic_training_steps=config.algo.num_critic_training_steps,
        num_pg_training_steps=config.algo.num_pg_training_steps,
        batch_size=config.algo.batch_size,
        replay_buffer_size=config.algo.replay_buffer_size,
        discount=config.algo.discount,
        reward_scaling=config.algo.reward_scaling,
        critic_learning_rate=config.algo.critic_learning_rate,
        actor_learning_rate=config.algo.actor_learning_rate,
        policy_learning_rate=config.algo.policy_learning_rate,
        noise_clip=config.algo.noise_clip,
        policy_noise=config.algo.policy_noise,
        soft_tau_update=config.algo.soft_tau_update,
        policy_delay=config.algo.policy_delay,
    )

    # Get the emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=config.algo.iso_sigma, line_sigma=config.algo.line_sigma
    )

    pg_emitter = PGAMEEmitter(
        config=pga_emitter_config,
        policy_network=policy_network,
        env=env,
        variation_fn=variation_fn,
    )

    # Instantiate MAP Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=pg_emitter,
        metrics_function=metrics_function,
    )

    # compute initial repertoire
    repertoire, emitter_state, random_key = map_elites.init(init_params, centroids, random_key)

    log_period = 10
    num_loops = int(config.num_iterations / log_period)

    metrics = dict.fromkeys(
        [
            "iteration", 
            "qd_score", 
            "coverage", 
            "max_fitness", 
            "time", 
            "evaluation", 
            "ga_offspring_added", 
            "qpg_offspring_added", 
            "ai_offspring_added"
            ], 
        jnp.array([])
        
        )
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )
    def plot_metrics_vs_iterations(metrics, log_period):
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

    # Main loop

    map_elites_scan_update = map_elites.scan_update
    eval_num = config.batch_size
    #print(f"Number of evaluations per iteration: {eval_num}")

    metrics_file_path = "./metrics_incremental.pickle"
    
    cumulative_time = 0
    for i in range(num_loops):
        start_time = time.time()
        (repertoire, emitter_state, random_key,), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, random_key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time
        cumulative_time += timelapse

        # Metrics


        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluation"] = jnp.arange(1+log_period*eval_num*i, 1+log_period*eval_num*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(cumulative_time, log_period)

        current_metrics["ga_offspring_added"], current_metrics["qpg_offspring_added"], current_metrics["ai_offspring_added"] = get_n_offspring_added(current_metrics)
        del current_metrics["is_offspring_added"]


        current_metrics_cpu = jax.tree_util.tree_map(lambda x: np.array(x), current_metrics)

        with open(metrics_file_path, "ab") as f:
            pickle.dump(current_metrics_cpu, f)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], current_metrics_cpu)
        log_metrics["qpg_offspring_added"] = np.sum(current_metrics["qpg_offspring_added"])
        log_metrics["ga_offspring_added"] = np.sum(current_metrics["ga_offspring_added"])
        log_metrics["ai_offspring_added"] = np.sum(current_metrics["ai_offspring_added"])
        csv_logger.log(log_metrics)


    # At the end, if you need one single combined structure, 
    # you can reload all increments and combine them:
    all_metrics = {}
    with open(metrics_file_path, "rb") as f:
        # Since we appended multiple chunks, read them all back
        metrics_list = []
        try:
            while True:
                m = pickle.load(f)
                metrics_list.append(m)
        except EOFError:
            pass

    # Combine all metrics arrays across the loaded chunks
    # This assumes all chunks have the same keys and shapes along axis=0
    for key in metrics_list[0].keys():
        all_metrics[key] = np.concatenate([m[key] for m in metrics_list], axis=0)

    # Now `all_metrics` contains all combined metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(all_metrics, metrics_file)

    # Repertoire
    os.mkdir("./repertoire/")
    #os.mkdir("./Plots/")
    repertoire.save(path="./repertoire/")
    
    #plot_metrics_vs_iterations(metrics, log_period)

    # Actor
    state_dict = serialization.to_state_dict(emitter_state.emitter_states[0].actor_params)
    with open("./actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Plot
    #if env.behavior_descriptor_length == 2:
        #env_steps = jnp.arange(config.num_iterations) * config.env.episode_length * config.batch_size
        #fig, _ = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_bd=config.env.min_bd, max_bd=config.env.max_bd)
        #fig.savefig("./Plots/repertoire_plot.png")

    #recreate_repertoire(repertoire, centroids, metrics_function, random_key)

if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
