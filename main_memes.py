import os

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple
import pickle

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.core.config_store import ConfigStore
from qdax.core.map_elites_memes import MAPElites
from qdax.types import RNGKey
from utils import Config, get_env
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from omegaconf import OmegaConf
from qdax.environments_v1 import behavior_descriptor_extractor
from qdax.core.neuroevolution.networks.networks import MLP



from qdax.core.containers.mapelites_repertoire_memes import compute_cvt_centroids, MapElitesRepertoire

from qdax.core.emitters.memes_emitter import MEMESConfig, MEMESEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition






@hydra.main(version_base="1.2", config_path="configs", config_name="memes")
def main(config: Config) -> None:


    # Choose stopping criteria
    if config.num_iterations > 0 and config.algo.num_evaluations > 0:
        print(
            "!!!WARNING!!! Both num_iterations and num_evaluations are set",
            "choosing num_iterations over num_evaluations",
        )
    if config.num_iterations > 0:
        num_iterations = config.num_iterations
    elif config.algo.num_evaluations > 0:
        num_iterations = (
            config.algo.num_evaluations
            // (
                config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps
            )
            + 1
        )
        

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)
    
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

    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    
    
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch_obs = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs)

    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print("Number of parameters in policy_network: ", param_count)
    
    @jax.jit
    def play_step_fn(
        env_state,
        policy_params,
        random_key):
        """
        Play an environment step and return the updated EnvState and the transition.

        Args: env_state: The state of the environment (containing for instance the
        actor joint positions and velocities, the reward...). policy_params: The
        parameters of policies/controllers. random_key: JAX random key.

        Returns:
            next_state: The updated environment state.
            policy_params: The parameters of policies/controllers (unchanged).
            random_key: The updated random key.
            transition: containing some information about the transition: observation,
                reward, next observation, policy action...
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return (next_state, policy_params, random_key), transition


    bd_extraction_fn = behavior_descriptor_extractor[config.env.name]
    scoring_fn = partial(
        scoring_function,
        episode_length=config.env.episode_length,
        play_reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    # Wrap the scoring function to do sampling


    # Define emitter
    es_emitter_config = MEMESConfig(
        sample_number=config.algo.sample_number,
        sample_sigma=config.algo.sample_sigma,
        sample_mirror=config.algo.sample_mirror,
        sample_rank_norm=config.algo.sample_rank_norm,
        num_in_optimizer_steps=config.algo.num_in_optimizer_steps,
        adam_optimizer=config.algo.adam_optimizer,
        learning_rate=config.algo.learning_rate,
        l2_coefficient=config.algo.l2_coefficient,
        novelty_nearest_neighbors=config.algo.novelty_nearest_neighbors,
        use_novelty_archive=config.algo.use_novelty_archive,
        use_novelty_fifo=config.algo.use_novelty_fifo,
        fifo_size=config.algo.fifo_size,
        proportion_explore=config.algo.proportion_explore,
        num_generations_stagnate=config.algo.num_generations_stagnate,
    )
    es_emitter = MEMESEmitter(
        config=es_emitter_config,
        batch_size=config.batch_size,
        scoring_fn=scoring_fn,
        num_descriptors=env.behavior_descriptor_length,
        scan_batch_size=config.algo.scan_batch_size,
        scan_novelty=config.algo.scan_novelty,
        total_generations=num_iterations,
        num_centroids=int(config.num_centroids),
    )
    
    reward_offset = 0
    
    metrics_fn = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=es_emitter,
        metrics_function=metrics_fn,
    )

    # Init algorithm
    repertoire, emitter_state, random_key = map_elites.init(
        init_params, centroids, random_key
    )
    
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
            ], 
        jnp.array([])
        )
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )

    metrics_file_path = "./metrics_incremental.pickle"

    # Main QD loop
    map_elites_scan_update = map_elites.scan_update
    eval_num = int((config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps) + config.batch_size)
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
        

        current_metrics["iteration"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluation"] = jnp.arange(1+log_period*eval_num*i, 1+log_period*eval_num*(i+1), dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(cumulative_time, log_period)
        current_metrics_cpu = jax.tree_util.tree_map(lambda x: np.array(x), current_metrics)

        with open(metrics_file_path, "ab") as f:
            pickle.dump(current_metrics_cpu, f)


        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], current_metrics_cpu)
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
    repertoire.save(path="./repertoire/")

    



if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="main", node=Config)
    main()
