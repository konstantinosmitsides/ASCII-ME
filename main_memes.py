import os

# Set environment variables to redirect cache directories
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
os.environ['WANDB_CACHE_DIR'] = '/tmp/wandb_cache'

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.map_elites_memes import MAPElites
from qdax.types import RNGKey
from qdax.utils.sampling import sampling
from utils import Config, get_env
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs as scoring_function
from qdax.utils.metrics import CSVLogger, default_qd_metrics



from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire

from core.emitters.memes_emitter import MEMESConfig, MEMESEmitter
from core.stochasticity_utils import reevaluation_function
from initialisation import set_up_default_metrics_dict, set_up_envs, set_up_metrics
from main_loop import main_loop
from qdax.core.neuroevolution.buffers.buffer import QDTransition






@hydra.main(version_base="1.2", config_path="configs", config_name="memes")
def train(config: Config) -> None:

    wandb.login(key="ab476069b53a15ad74ff1845e8dee5091d241297")
    wandb.init(
        project="me-mcpg",
        name=config.algo.name,
        config=OmegaConf.to_container(config, resolve=True),
    )

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
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
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
        num_descriptors=config.env.behavior_descriptor_length,
        scan_batch_size=config.algo.scan_batch_size,
        scan_novelty=config.algo.scan_novelty,
        total_generations=num_iterations,
        num_centroids=int(config.num_centroids),
    )
    
    metrics_fn = partial(
        default_qd_metrics,
        qd_offset=reward_offset * config.env.episode_length,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=me_scoring_fn,
        emitter=es_emitter,
        metrics_function=metrics_fn,
    )

    # Init algorithm
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    
    log_period = 10
    num_loops = int(config.num_iterations / log_period)
    metrics = dict.fromkeys(
        [
            "iteration", 
            "qd_score", 
            "coverage", 
            "max_fitness", 
            #"qd_score_repertoire", 
            #"dem_repertoire", 
            "time", 
            "evaluation", 
            #"ga_offspring_added", 
            #"qpg_offspring_added"
            ], 
        jnp.array([])
        )
    csv_logger = CSVLogger(
        "./log.csv",
        header=list(metrics.keys())
    )


    # Main QD loop
    map_elites_scan_update = map_elites.scan_update
    eval_num = config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps
    cumulative_time = 0
    
    for i in range(num_loops):
        #print(f"Loop {i+1}/{num_loops}")
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
        #current_metrics["qd_score_repertoire"] = jnp.repeat(qd_score_repertoire, log_period)
        #current_metrics["dem_repertoire"] = jnp.repeat(dem_repertoire, log_period)
        #current_metrics["ga_offspring_added"], current_metrics["qpg_offspring_added"] = get_n_offspring_added(current_metrics)
        #del current_metrics["is_offspring_added"]
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        #log_metrics["qpg_offspring_added"] = jnp.sum(current_metrics["qpg_offspring_added"])
        #log_metrics["ga_offspring_added"] = jnp.sum(current_metrics["ga_offspring_added"])
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)
    #profiler.stop_trace()
    # Metrics
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)

    # Repertoire
    os.mkdir("./repertoire/")
    #os.mkdir("./Plots/")
    repertoire.save(path="./repertoire/")

    



if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
