from typing import Tuple
from dataclasses import dataclass
import pickle

import pandas as pd

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax import environments_v1, environments
from qdax.core.neuroevolution.networks.networks import MLP, MLPMCPG
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from functools import partial

from omegaconf import OmegaConf
from functools import partial


@dataclass
class Config:
    name: str
    seed: int
    num_iterations: int
    batch_size: int

    # Archive
    num_init_cvt_samples: int
    num_centroids: int
    policy_hidden_layer_sizes: Tuple[int, ...]

@dataclass
class ConfigReproducibility:
    env_name: str
    algo_name: str
    num_evaluations: int


def get_env(config: Config):
    if config.env.version == "v1":
        if config.env.name == "hopper_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "walker2d_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "halfcheetah_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length)
        elif config.env.name == "ant_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "ant_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "anttrap_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation)
        elif config.env.name == "humanoid_uni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
            )
                #backend=config.env.backend)
        elif config.env.name == "humanoid_omni":
            env = environments_v1.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                #backend=config.env.backend
            )
        else:
            raise ValueError("Invalid environment name.")
    elif config.env.version == "v2":
        if config.env.name == "hopper_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "walker2d_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "halfcheetah_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                backend=config.env.backend)
        elif config.env.name == "ant_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "ant_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "anttrap_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                use_contact_forces=config.env.use_contact_forces,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "humanoid_uni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        elif config.env.name == "humanoid_omni":
            env = environments.create(
                config.env.name,
                episode_length=config.env.episode_length,
                exclude_current_positions_from_observation=config.env.exclude_current_positions_from_observation,
                backend=config.env.backend)
        else:
            raise ValueError("Invalid environment name.")
    else:
        raise ValueError("Invalid Brax version.")

    return env


def get_config(run_dir):
    config = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    return config

def get_metrics(run_dir):
    with open(run_dir / "metrics.pickle", "rb") as metrics_file:
        metrics = pickle.load(metrics_file)
        if "ga_offspring_added" in metrics.keys():
            if metrics["ga_offspring_added"].shape[0] == metrics["iteration"].shape[0] // 10:
                metrics["ga_offspring_added"] = jnp.tile(metrics["ga_offspring_added"], 10)
        if "qpg_offspring_added" in metrics.keys():
            if metrics["qpg_offspring_added"].shape[0] == metrics["iteration"].shape[0] // 10:
                metrics["qpg_offspring_added"] = jnp.tile(metrics["qpg_offspring_added"], 10)
        if metrics["evaluation"].shape[0] != metrics['iteration'].shape[0]:    
            del metrics["evaluation"]
    return pd.DataFrame.from_dict(metrics)

def get_log(run_dir):
    return pd.read_csv(run_dir / "log.csv")

def get_repertoire(run_dir):
    # Get config
    config = get_config(run_dir)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init environment
    env = get_env(config)

    # Init policy network
    
    if config.algo.name == "mcpg_me":
        policy_network = MLPMCPG(
        action_dim=env.action_size,
        activation=config.algo.activation,
        no_neurons=config.algo.no_neurons,
    )
    else:
        
        policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )
    

    # Init fake params
    random_key, random_subkey = jax.random.split(random_key)
    fake_obs = jnp.zeros(shape=(env.observation_size,))
    fake_params = policy_network.init(random_subkey, fake_obs)

    # Load repertoire
    _, reconstruction_fn = ravel_pytree(fake_params)

    # Return repertoire
    return MapElitesRepertoire.load(reconstruction_fn=reconstruction_fn, path=str(run_dir) + "/repertoire/")

def get_df(results_dir):
    metrics_list = []
    for env_dir in results_dir.iterdir():
        if env_dir.is_file() or env_dir.name not in ["ant_omni", "anttrap_omni", "walker2d_uni","ant_uni", "hopper_uni"]:
            continue        
            
        
        print(env_dir.name)
        for algo_dir in env_dir.iterdir():
            for run_dir in algo_dir.iterdir():
                
                # Get config and metrics
                
                config = get_config(run_dir)
                metrics = get_metrics(run_dir)
                

                metrics["env"] = env_dir.name
                

                # Algo
                metrics["algo"] = config.algo.name
                if config.algo.name != "ppga":
                    if config.algo.name == "memes":
                        metrics["batch_size"] = config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps
                    else:
                        metrics["batch_size"] = config.batch_size
                else:
                    metrics["batch_size"] = config.algo.env_batch_size


                if config.algo.name == "dcrl_me" or config.algo.name == "pga_me":
                    metrics["num_critic_training_steps"] = config.algo.num_critic_training_steps
                    metrics["num_pg_training_steps"] = config.algo.num_pg_training_steps
                    metrics["training_batch_size"] = config.algo.batch_size
                    
                if config.algo.name == "ascii_me":
                    metrics["proportion_mutation_ga"] = config.algo.proportion_mutation_ga
                    metrics["no_epochs"] = config.algo.no_epochs
                    metrics['greedy'] = config.algo.greedy



                # Run
                metrics["run"] = run_dir.name

                # Number of Evaluations

                if config.algo.name == "ppga":
                    metrics["num_evaluations"] = metrics["evaluation"]
                elif config.algo.name == "memes":
                    metrics["num_evaluations"] = metrics["iteration"] * ((config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps) + config.batch_size)
                else:
                    metrics["num_evaluations"] = metrics["iteration"] * config.batch_size

                # Coverage
                metrics["coverage"] /= 100

                metrics_list.append(metrics)
    return pd.concat(metrics_list, ignore_index=True)


def flatten_policy_parameters(params, flat_params=None):
    if flat_params is None:
        flat_params = []
    
    for key, value in params.items():
        if isinstance(value, dict):
            # Recursive call to handle nested dictionaries
            flatten_policy_parameters(value, flat_params)
        elif key in ['bias', 'kernel', 'log_std']:
            # Flatten and append the parameters if they match expected keys
            #print(f"Including {key} with shape {value.shape}")  # Debug: Confirm these parameters are included
            flat_params.append(jnp.ravel(value))
        else:
            continue
            #print(f"Skipping {key}")  # Debug: Notice skipped params or incorrect structures

    return jnp.concatenate(flat_params) if flat_params else jnp.array([])


def calculate_gae(data):
    def _scan_get_advantages(carry, x):
        gae, next_value = carry
        done, value, reward = x
        
        
        delta = reward + 0.99 * next_value * (1 - done) - value
        
        gae = delta + 0.99 * 0.95 * (1 - done) * gae
        
        return (gae, value), gae
    
    _, advantages = jax.lax.scan(
        _scan_get_advantages,
        (jnp.zeros_like(data.val_adv[:, 0]), data.target[:, -1]),
        (data.dones.T, data.val_adv.T, data.rewards.T),
        reverse=True,
        unroll=16,
    )
    
    return advantages.T, advantages.T + data.val_adv
    
    
@jax.jit
def transfer_params(target_params, source_params):
    source_params_ = source_params['params']
    target_params_ = target_params['params']
    target_params_['Dense_0']['kernel'] = source_params_['Dense_0']['kernel']
    target_params_['Dense_0']['bias'] = source_params_['Dense_0']['bias']
    target_params_['Dense_1']['kernel'] = source_params_['Dense_1']['kernel']
    target_params_['Dense_1']['bias'] = source_params_['Dense_1']['bias']
    target_params_['Dense_2']['kernel'] = source_params_['Dense_2']['kernel']
    target_params_['Dense_2']['bias'] = source_params_['Dense_2']['bias']
    target_params_['log_std'] = source_params_['log_std']
    target_params['params'] = target_params_
    
    return target_params

@jax.jit
def transfer_params_no_pg(target_params, source_params):
    source_params_ = source_params['params']
    target_params_ = target_params['params']
    target_params_['Dense_0']['kernel'] = source_params_['Dense_0']['kernel']
    target_params_['Dense_0']['bias'] = source_params_['Dense_0']['bias']
    target_params_['Dense_1']['kernel'] = source_params_['Dense_1']['kernel']
    target_params_['Dense_1']['bias'] = source_params_['Dense_1']['bias']
    target_params_['Dense_2']['kernel'] = source_params_['Dense_2']['kernel']
    target_params_['Dense_2']['bias'] = source_params_['Dense_2']['bias']
    #target_params_['log_std'] = source_params_['log_std']
    target_params['params'] = target_params_
    
    return target_params

@jax.jit
def normalize_obs(obs, mean, var):
    return (obs - mean) / jnp.sqrt(var + 1e-8)
    
@partial(jax.jit, static_argnames=("episode_length",))
def get_return_for_episode(
    rewards,
    episode_length,
):
    def _body(carry, x):
        (next_return,) = carry
        (rewards,) = x

        current_return = rewards + 0.99 * next_return
        return (current_return,), (current_return,)
    
    
    
    #jax.debug.print("rewards", rewards.shape)
    
    _, (return_,) = jax.lax.scan(
        _body,
        (jnp.array(0.),),
        (rewards,),
        length=episode_length,
        reverse=True,
    )
    
    return return_


@partial(jax.jit, static_argnames=("episode_length",))
def get_return_for_batch_episodes(
    rewards,
    mask,
    episode_length
):
    #mask = jnp.expand_dims(mask, axis=-1)
    #valid_rewards = (rewards * mask)#.squeeze(axis=-1)
    #jax.debug.print("mask: {}", mask.shape)
    #jax.debug.print("rewards*mask: {}", (rewards * mask).shape)
    return jax.vmap(get_return_for_episode, in_axes=(0, None))(rewards * mask, episode_length)


@jax.jit
def concatenate_params(param_dict):
    concatenated_arrays = []
    for layer in param_dict['params']:
        if layer == 'log_std':
            continue
        layer_data = param_dict['params'][layer]
        # Extract and flatten the arrays if necessary
        bias = layer_data['bias']
        kernel = layer_data['kernel']
        kernel = kernel.reshape(kernel.shape[0], -1)
        # Concatenate bias and kernel horizontally
        concatenated = jnp.concatenate((bias, kernel), axis=1)
        concatenated_arrays.append(concatenated)
    # Concatenate all layers vertically if needed
    final_array = jnp.concatenate(concatenated_arrays, axis=1)
    return final_array


@jax.jit
def find_magnitude_of_updates(params, new_params):
    diff = new_params - params
    return jnp.linalg.norm(diff, axis=1)

@jax.jit
def compute_cosine_similarity(states1, states2):
    """
    Compute the cosine similarity between corresponding pairs of states in two arrays.

    Parameters:
    - states1: JAX array of shape (N, D), where N is the number of states and D is the state dimension.
    - states2: JAX array of shape (N, D), must have the same shape as states1.

    Returns:
    - similarities: JAX array of shape (N,), containing the cosine similarities.
    """
    # Compute the dot product between corresponding states
    dot_products = jnp.sum(states1 * states2, axis=1)
    
    # Compute the norms (magnitudes) of each state vector
    norms1 = jnp.linalg.norm(states1, axis=1)
    norms2 = jnp.linalg.norm(states2, axis=1)
    
    # Compute the product of norms
    norm_products = norms1 * norms2
    
    # Handle cases where norms are zero to avoid division by zero
    # Use jnp.where to avoid division by zero
    safe_norm_products = jnp.where(norm_products == 0, 1.0, norm_products)
    
    # Compute cosine similarities
    cosine_similarities = dot_products / safe_norm_products
    
    # Set similarities to zero where norms were zero
    cosine_similarities = jnp.where(norm_products == 0, 0.0, cosine_similarities)
    
    # Set negative similarities to zero
    cosine_similarities = jnp.maximum(cosine_similarities, 0.25)

    return cosine_similarities