from functools import partial
import os
import time
import pickle
from absl import logging

import jax
import jax.numpy as jnp

from qdax.environments import behavior_descriptor_extractor, reward_offset
from qdax.core.map_elites import MAPElites
from qdax.tasks.brax_envs import scoring_function_brax_envs as scoring_function
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids # utility for computing centroids in a CVT; it is a method used in ME for dividing the search space
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
    env = get_env(config) # it creates the environment by connecting using the functionfrom the utils.py file and calling the env key from config 'me'
    #reset_fn = jax.jit(env.reset) # FROM WHERE DO THEY ACCESS .RESET?
    
    #def scoring_fn(genotypes, key):
    #    pass
    
    # Compute the centroids
    centroids, key = compute_cvt_centroids(
		num_descriptors=env.behavior_descriptor_length,  # FROM WHERE DO THEY ACCESS BEHAVIOR_DESCRIPTOR_LENGTH?
		num_init_cvt_samples=config.qd.num_init_cvt_samples,  # WHY IS IT QD ONLY HERE?
		num_centroids=config.qd.num_centroids,
		minval=config.env.min_bd,
		maxval=config.env.max_bd,
		random_key=key,
	)
    
    # Init policy net
    policy_layers_sizes = config.qd.policy_hidden_layer_sizes + (env.action_size, )  # WHERE CAN I SEE THE ACTION_SIZE
    policy_network = MLP(
        layer_sizes=policy_layers_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),   # how to initialize the weights of the layers
        fina_activation=jnp.tanh,
    )
    
    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)    # splits random_key into 2 new random keys
    keys = jax.random.split(subkey, num=config.batch_size)   # subkey is split into config.batch_size new keys stored in the array keys; this provides a unique random key for each example in the batch 
    fake_batch_obs = jnp.zeros(shape=(config.batch_size, env.observation_size))  # creates a batch of fake observations, which are all initialized to zero.
    
    # Initializes parameters separately for each observation in the batch of fake observations using the corresponding key from keys. 
    # Each element of the bacth gets is own set of initialized parameters independent of others, which is important 
    # for initializing different instances of a model in parallel during multi-agent simulations
    init_params = jax.vmap(policy_network.init)(keys, fake_batch_obs) 
    
    # Create the initial environment states
    random_key, subkey = jax.random.split(random_key)
    keys = jnp.repeat(jnp.expand_dims(subkey, axis=0), repeats=config.batch_size, axis=0)
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)
    
    param_count = sum(x[0].size for x in jax.tree_util.tree_leaves(init_params))
    print(f"Number of parameters in policy_network: {param_count}")
    
    # Define the function to play a step with the policy in the environment
    def play_step_fn(env_state, policy_params, random_key):
        
        """
        Play an environement step and return the updated state and the transition.
        
        Args:
            env_state: current state
            policy_params: corresponding policy parameters
            random_key: corresponding random key
            
        Returns:
            next_state: next state after policy has acted
            policy_params: corresponding policy parameters
            random_key: corresponding random key
            transition: all the info from QDTransition
        """
        
        actions = policy_network.apply(policy_params, env_state.obs)  # given the current state, findthe actions you take
        state_desc = env_state.info["state_descriptor"]  # get the descriptor of the current state
        next_state = env.step(env_state, actions)
        
        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,  # WHAT IS THIS?
            truncations=next_state.info["truncation"],  # indicates if an episode has reached max time step
            actions=actions,
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,   # WHAT IS THIS?
            desc_prime=jnp.zeros(env.behavior_descriptor_length, ) * jnp.nan,   # WHAT IS THIS?
        )
        
        return next_state, policy_params, random_key, transition
    
    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[config.qd.env.name]
    scoring_fn = partial(
        scoring_function,
        init_states=init_states,
        episode_length=config.env.episode_length,
        play_step_fn=play_step_fn,  # WHAT EXACLTY HAPPENS HERE? WHAT IS THE INPUT OF PLAY_STEP_FN?
        behavior_descriptor_extractor=bd_extraction_fn,
    )
    
    @jax.jit
    def evaluate_repertoire(random_key, repertoire):
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        
        fitnesses, descriptors, extra_scores, random_key = scoring_fn(
            repertoire.genotypes, random_key
        )
        
        # Compute reperoire QD score
        qd_score = jnp.sum((1.0 - repertoire_empty) * fitnesses).astype(float)
        
        # Compute repertoire desc error mean
        error = jnp.linalg.norm(repertoire.descrpitors - descriptors, axis=1)
        dem = (jnp.sum((1.0 - repertoire_empty) * error) / jnp.sum(1.0 - repertoire_empty)).astype(float)
        
        return random_key, qd_score, dem
    
    def get_elites(metric):
        return jnp.sum(metric, axis=-1)
    
    # Get a minimum reward value to make sure qs_score are positive 
    reward_offset = reward_offset[config.qd.0env.name]
    
    # Define a metrics function
    metrics_fn = partial(
		default_qd_metrics,
		qd_offset=reward_offset * config.env.episode_length,  # TODO
	)
    
    # Define emitter
    # WHAT IS THE INPUT OF ISOLINE_VARIATION?
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
