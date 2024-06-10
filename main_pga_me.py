import os

# Set environment variables to redirect cache directories
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['WANDB_CACHE_DIR'] = '/tmp/wandb_cache'

from typing import Tuple
from dataclasses import dataclass
import functools
import os
import time
import pickle

import jax
import jax.numpy as jnp
from flax import serialization

from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.environments import behavior_descriptor_extractor
from qdax.core.map_elites_pga import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEConfig, PGAMEEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_map_elites_results
from set_up_brax import get_reward_offset_brax


import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import wandb
from utils import Config, get_env
from jax.lib import xla_bridge


@hydra.main(version_base="1.2", config_path="configs/", config_name="pga_me")
def main(config: Config) -> None:
    wandb.init(
        project="DCG-MAP-Elites",
        name=config.algo.name,
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
    print(type(env.action_size))
    print(type(config.policy_hidden_layer_sizes))
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
            #desc=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
            #desc_prime=jnp.zeros(env.behavior_descriptor_length,) * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = functools.partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    reward_offset = get_reward_offset_brax(env, config.env.name)
    print(f"Reward offset: {reward_offset}")
    
    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf

        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )

        # Compute repertoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)
        qd_score += reward_offset * config.env.episode_length * jnp.sum(1.0 - repertoire_empty)

        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descriptors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)

        return random_key, qd_score, dem

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

    # Get minimum reward value to make sure qd_score are positive
    #reward_offset = 0


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
        #actor_learning_rate=config.algo.actor_learning_rate,
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

    metrics = dict.fromkeys(["iteration", "qd_score", "coverage", "max_fitness", "qd_score_repertoire", "dem_repertoire", "actor_fitness", "time"], jnp.array([]))
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    # Main loop
    map_elites_scan_update = map_elites.scan_update
    for i in range(num_loops):
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
        random_key, fitness_actor = evaluate_actor(random_key, emitter_state.emitter_states[0].actor_params)

        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        current_metrics["qd_score_repertoire"] = jnp.repeat(qd_score_repertoire, log_period)
        current_metrics["dem_repertoire"] = jnp.repeat(dem_repertoire, log_period)
        current_metrics["actor_fitness"] = jnp.repeat(fitness_actor, log_period)
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
    repertoire.save(path="./repertoire/")

    # Actor
    state_dict = serialization.to_state_dict(emitter_state.emitter_states[0].actor_params)
    with open("./actor.pickle", "wb") as params_file:
        pickle.dump(state_dict, params_file)

    # Plot
    if env.behavior_descriptor_length == 2:
        env_steps = jnp.arange(config.num_iterations) * config.env.episode_length * config.batch_size
        fig, _ = plot_map_elites_results(env_steps=env_steps, metrics=metrics, repertoire=repertoire, min_bd=config.env.min_bd, max_bd=config.env.max_bd)
        fig.savefig("./plot.png")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
