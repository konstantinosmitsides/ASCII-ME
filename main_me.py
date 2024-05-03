from functools import partial
import os
import time
import pickle
from absl import logging

import jax
import jax.numpy as jnp

from qdax.environments import behavior_descriptor_extractor
from qdax.core.map_elites import MAPElites
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.map_elites import MAPElites
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from analysis.plot_repertoire import plot_repertoire
from utils import get_metric, get_env

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
from utils import get_env
import matplotlib.pyplot as plt


@hydra.main(version_base=None, config_path="configs/", config_name="me")
def main(config: DictConfig) -> None:
    logging.info("Starting MAP-Elites...")
    wandb.init(
		project="ME-WITH-SAMPLE-BASED-DRL",
		name=config.qd.algo,
		config=OmegaConf.to_container(config, resolve=True),
	)
    
    os.mkdir("./repertoires/")
    
    # Init a random key
    key = jax.random.PRNGKey(config.seed)
    
    # Init env
    logging.info("Initializing env...")
    env = get_env(config)
    reset_fn = jax.jit(env.reset)
    
    def scoring_fn(genotypes, key):
        pass
    
    # Compute the centroids
    centroids, key = compute_cvt_centroids(
		num_descriptors=min_bd.size,
		num_init_cvt_samples=config.qd.num_init_cvt_samples,
		num_centroids=config.qd.num_centroids,
		minval=min_bd,
		maxval=max_bd,
		random_key=key,
	)
    
    # Define a metrics function
    metrics_fn = partial(
		default_qd_metrics,
		qd_offset=0.,  # TODO
	)
    
    # Define emitter
    variation_fn = partial(
		isoline_variation, iso_sigma=config.qd.iso_sigma, line_sigma=config.qd.line_sigma,
	)
    
    mixing_emitter = MixingEmitter(
		mutation_fn=None,
		variation_fn=variation_fn,
		variation_percentage=1.0,
		batch_size=config.qd.batch_size
	)
    
    # Instantiate MAP-Elites
    map_elites = MAPElites(
		scoring_function=scoring_fn,
		emitter=mixing_emitter,
		metrics_function=metrics_fn,
	)
    
    # Compute initial repertoire and emitter state
    logging.info("Initializing MAP-Elites...")
    key, subkey = jax.random.split(key)
    init_params = 0  # TODO
    repertoire, emitter_state, key = map_elites.init(init_params, centroids, key)
    
    metrics = dict.fromkeys(["generation", "qd_score", "coverage", "max_fitness", "time"], jnp.array([]))
    csv_logger = CSVLogger(
		"./log.csv",
		header=list(metrics.keys())
	)
    
    # Main loop
    logging.info("Starting main loop...")
    map_elites_scan_update = map_elites.scan_update
    for i, generation in enumerate(range(0, config.qd.num_generations, config.qd.generation_period)):
        start_time = time.time()
        (repertoire, emitter_state, key,), current_metrics = jax.lax.scan(
			map_elites_scan_update,
			(repertoire, emitter_state, key),
			(),
			length=config.qd.generation_period,
		)
        timelapse = time.time() - start_time
        
        variance_repertoire = jnp.mean(jnp.var(repertoire.observations, axis=0, where=(repertoire.fitnesses != -jnp.inf)[:, None, None, None]))
        
        # Metrics
        current_metrics["generation"] = jnp.arange(1+generation, 1+generation+config.qd.generation_period, dtype=jnp.int32)
        current_metrics["time"] = jnp.repeat(timelapse, config.qd.generation_period)
        metrics = jax.tree_util.tree_map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)
        
        # Log
        log_metrics = jax.tree_util.tree_map(lambda metric: metric[-1], metrics)
        csv_logger.log(log_metrics)
        wandb.log(log_metrics)
        logging.info(log_metrics)
        
        # Repertoire
        if i % config.qd.log_period == 0:
            logging.info("Saving repertoire...")
            os.mkdir(f"./repertoires/repertoire_{generation}/")
            repertoire.replace(observations=jnp.nan).save(path=f"./repertoires/repertoire_{generation}/")
            
            # Plot
            if min_bd.size == 2:
                fig, _ = plot_repertoire(config, repertoire)
                fig.savefig("repertoire.pdf", bbox_inches="tight")
                plt.close()
                
    # Metrics
    logging.info("Saving metrics...")
    with open("./metrics.pickle", "wb") as metrics_file:
        pickle.dump(metrics, metrics_file)
        
    # Repertoire    
    logging.info("Saving repertoire...")
    os.mkdir("./repertoires/repertoire/")
    repertoire.replace(observations=jnp.nan).save(path="./repertoires/repertoire/")
    
    # Plot
    if min_bd.size == 2:
        fig, _ = plot_repertoire(config, repertoire)
        fig.savefig("repertoire.pdf", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
	main()
