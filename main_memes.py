import os
import logging
import time
import pickle
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple, Callable, List

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.map_elites import MAPElites
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids, MapElitesRepertoire, compute_cvt_centroids
from qdax.types import RNGKey, Genotype
from qdax.utils.sampling import sampling
from qdax.core.neuroevolution.networks.networks import MLP
#from initialisation import set_up_envs as set_up_envs_local

from set_up_brax import (
    get_behavior_descriptor_length_brax,
    get_environment_brax,
    get_policy_struc_brax,
    get_reward_offset_brax,
    get_scoring_function_brax,
)

from qdax.core.emitters.memes_emitter import (
    MEMESFixResetConfig,
    MEMESFixResetEmitter,
)

@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    alg_name: str
    # Env config
    seed: int
    env_name: str
    episode_length: int
    policy_hidden_layer_sizes: Tuple[int, ...]
    # ME config
    num_evaluations: int
    num_iterations: int
    batch_size: int
    num_samples: int
    fixed_init_state: bool
    discard_dead: bool
    # Grid config
    grid_shape: Tuple[int, ...]
    # Emitter config
    iso_sigma: float
    line_sigma: float
    crossover_percentage: float
    # others
    log_period: int  # only for timings and metrics
    store_repertoire: bool
    store_repertoire_log_period: int

    # ES parameters
    sample_number: int
    sample_sigma: float
    sample_mirror: bool
    sample_rank_norm: bool
    num_generations_sample: int
    adam_optimizer: bool
    learning_rate: float
    l2_coefficient: float  # coefficient for regularisation
    novelty_nearest_neighbors: int
    use_novelty_archive: bool  # if 1, use repertoire for novelty
    use_novelty_fifo: bool  # if 1 use fifo archive for novelty
    fifo_size: int  # size of the fifo buffer
    proportion_explore: bool

    scan_batch_size: int
    scan_novelty: int

def set_up_envs(
    config: ConfigStore,
    batch_size: int,
    random_key: RNGKey,
) -> Tuple[Any, Callable, Any, Callable, Genotype, float, int, int, RNGKey]:

    # Init environment and population of controllers
    print("Env name: ", config.env_name)

    # Initialising environment
    env = get_environment_brax(
        config.env_name, config.episode_length, config.fixed_init_state
    )

    # Get network size
    input_size, output_size, policy_layer_sizes, activation = get_policy_struc_brax(
        env, config.policy_hidden_layer_sizes
    )

    # Create the network
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=activation,
    )

    # Get the scoring function
    scoring_fn, random_key = get_scoring_function_brax(
        env,
        config.env_name,
        config.episode_length,
        policy_network,
        random_key,
    )

    # Build init variables
    def construction_fn(size: int, random_key: RNGKey) -> jnp.ndarray:
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=size)
        fake_batch = jnp.zeros(shape=(size, output_size))
        init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
        return init_variables, random_key

    # Build all common parts
    reward_offset = get_reward_offset_brax(env, config.env_name)
    behavior_descriptor_length = get_behavior_descriptor_length_brax(
        env, config.env_name
    )
    init_variables, random_key = construction_fn(batch_size, random_key)
    genotype_dim = jnp.prod(jnp.asarray(config.policy_hidden_layer_sizes))

    return (
        env,
        scoring_fn,
        policy_network,
        construction_fn,
        init_variables,
        reward_offset,
        behavior_descriptor_length,
        genotype_dim,
        random_key,
    )

@hydra.main(version_base=None, config_path="configs", config_name="memes")
def train(config: ExperimentConfig) -> None:

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    # Choose stopping criteria
    if config.num_iterations > 0 and config.num_evaluations > 0:
        print(
            "!!!WARNING!!! Both num_iterations and num_evaluations are set",
            "choosing num_iterations over num_evaluations",
        )
    if config.num_iterations > 0:
        num_iterations = config.num_iterations
    elif config.num_evaluations > 0:
        num_iterations = (
            config.num_evaluations
            // (
                config.batch_size * config.sample_number
            )
            + 1
        )

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Setup environment
    
    (
        env,
        scoring_fn,
        policy_network,
        construction_fn,
        init_variables,
        reward_offset,
        behavior_descriptor_length,
        genotype_dim,
        random_key,
    ) = set_up_envs(config, config.batch_size, random_key)


    # Wrap the scoring function to do sampling
    me_scoring_fn = partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=config.num_samples,
    )

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    '''
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids = compute_euclidean_centroids(
        grid_shape=config.grid_shape,
        minval=minval,
        maxval=maxval,
    )
    '''
    init_time = time.time()
    
    minval, maxval = env.behavior_descriptor_limits
    centroids, _ = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=minval,
        maxval=maxval,
        random_key=random_key,
    )

    centroid_time = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {centroid_time:.2f}s")

    # Define emitter
    es_emitter_config = MEMESFixResetConfig(
        sample_number=config.sample_number,
        sample_sigma=config.sample_sigma,
        sample_mirror=config.sample_mirror,
        sample_rank_norm=config.sample_rank_norm,
        num_generations_sample=config.num_generations_sample,
        adam_optimizer=config.adam_optimizer,
        learning_rate=config.learning_rate,
        l2_coefficient=config.l2_coefficient,
        novelty_nearest_neighbors=config.novelty_nearest_neighbors,
        use_novelty_archive=config.use_novelty_archive,
        use_novelty_fifo=config.use_novelty_fifo,
        fifo_size=config.fifo_size,
        proportion_explore=config.proportion_explore,
    )
    es_emitter = MEMESFixResetEmitter(
        config=es_emitter_config,
        batch_size=config.batch_size,
        scoring_fn=scoring_fn,
        num_descriptors=env.behavior_descriptor_length,
        scan_batch_size=config.scan_batch_size,
        scan_novelty=config.scan_novelty,
        total_generations=num_iterations,
        num_centroids=int(jnp.prod(jnp.array(config.grid_shape))),
    )

    # Define metrics functions
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        min_fitness = jnp.min(
            jnp.where(repertoire.fitnesses > -jnp.inf, repertoire.fitnesses, jnp.inf)
        )
        return {
            "qd_score": jnp.array([qd_score]),
            "max_fitness": jnp.array([max_fitness]),
            "min_fitness": jnp.array([min_fitness]),
            "coverage": jnp.array([coverage]),
        }

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=me_scoring_fn,
        emitter=es_emitter,
        metrics_function=metrics_fn,
    )

    # Init algorithm
    logger.warning("--- Algorithm initialisation ---")
    start_time = time.time()
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    init_time = time.time() - start_time
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")

    # Set up metric dicts
    full_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "coverage": jnp.array([0.0]),
        "max_fitness": jnp.array([0.0]),
        "min_fitness": jnp.array([0.0]),
        "qd_score": jnp.array([0.0]),
    }
    timings = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "init_time": init_time,
        "centroids_time": centroid_time,
        "runtime_logs": jnp.zeros([(num_iterations) + 1, 1]),
        "avg_iteration_time": 0.0,
    }

    # Function to count number of evaluations
    count_evals_fn = (
        lambda iteration: iteration
        * config.batch_size
        * config.sample_number
    )

    # All necessary functions and parameters
    map_elites_update_fn = partial(map_elites.update)
    log_period = config.log_period
    store_repertoire = config.store_repertoire
    store_repertoire_log_period = config.store_repertoire_log_period
    episode_length = config.episode_length
    output_dir = "./"

    # Setup metrics checkpoint save
    _last_metrics_dir = os.path.join(output_dir, "checkpoints", "last_metrics")
    os.makedirs(_last_metrics_dir, exist_ok=True)
    _grid_img_dir = os.path.join(output_dir, "images", "me_grids")
    os.makedirs(_grid_img_dir, exist_ok=True)
    _metrics_img_dir = os.path.join(output_dir, "images", "me_metrics")
    os.makedirs(_metrics_img_dir, exist_ok=True)
    _timings_dir = os.path.join(output_dir, "timings")
    os.makedirs(_timings_dir, exist_ok=True)

    # Setup repertoire checkpoint save
    _last_grid_dir = os.path.join(output_dir, "checkpoints", "last_grid")
    os.makedirs(_last_grid_dir, exist_ok=True)

    # Main QD Loop
    total_start_time = time.time()
    algorithm_time = 0.0
    total_evals = 0
    for iteration in range(num_iterations):
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_iterations} ---"
        )

        start_time = time.time()
        (
            repertoire,
            emitter_state,
            metrics,
            random_key,
        ) = map_elites_update_fn(
            repertoire,
            emitter_state,
            random_key,
        )
        iteration_time = time.time() - start_time
        algorithm_time += iteration_time

        logger.warning(f"--- Current QD Score: {metrics['qd_score'][-1]:.2f}")
        logger.warning(f"--- Current Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning(f"--- Current Max Fitness: {metrics['max_fitness'][-1]}")
        logger.warning(f"--- Iteration time: {iteration_time}")

        # Add epoch and evals
        total_evals += count_evals_fn(iteration)
        metrics["epoch"] = jnp.array([iteration])
        metrics["evals"] = jnp.array([total_evals])
        metrics["time"] = jnp.array([algorithm_time])

        # Save metrics
        full_metrics = {
            key: jnp.concatenate((full_metrics[key], metrics[key]))
            for key in full_metrics
        }
        if iteration % log_period == 0:
            with open(
                os.path.join(_last_metrics_dir, "metrics.pkl"), "wb"
            ) as file_to_save:
                pickle.dump(full_metrics, file_to_save)

        # Store the latest controllers of the repertoire
        if store_repertoire and iteration % store_repertoire_log_period == 0:
            repertoire.save(path=_last_grid_dir + "/")

    total_time = time.time() - total_start_time
    logger.warning("--- Final metrics ---")
    logger.warning(f"Total time: {total_time:.2f}s")
    logger.warning(f"Algorithm time: {algorithm_time:.2f}s")
    logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")

    # Save final metrics
    with open(os.path.join(_last_metrics_dir, "metrics.pkl"), "wb") as file_to_save:
        pickle.dump(full_metrics, file_to_save)
    # Save final repertoire
    repertoire.save(path=_last_grid_dir + "/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
