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
                backend=config.env.backend)
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

def get_df(results_dir, episode_length):
    metrics_list = []
    for env_dir in results_dir.iterdir():
        if env_dir.is_file() or env_dir.name not in ["ant_omni_250", "anttrap_omni_250", "humanoid_omni", "walker2d_uni_250","walker2d_uni_1000", "halfcheetah_uni", "ant_uni_250", "ant_uni_1000", "humanoid_uni", "hopper_uni_250", "hopper_uni_1000"]:
            continue        
            continue
        if env_dir.name[-3:] != str(episode_length)[-3:]:
            continue
        
        print(env_dir.name)
        for algo_dir in env_dir.iterdir():
            for run_dir in algo_dir.iterdir():
                if run_dir.name[:10] == "2024-08-24":
                    continue
                
                
                
                # Get config and metrics
                
                config = get_config(run_dir)
                metrics = get_metrics(run_dir)
                
                if config.algo.name == "mcpg_me":
                    if run_dir.name[:17] != "2024-08-29_200827":
                #        print('continue')
                        continue

                # Env
                metrics["env"] = f"{config.env.name}_{episode_length}"
                

                # Algo
                metrics["algo"] = config.algo.name
                metrics["batch_size"] = config.batch_size

                # Run
                metrics["run"] = run_dir.name

                # Number of Evaluations
                if config.algo.name == "me_es":
                    metrics["num_evaluations"] = metrics["iteration"] * 1050
                elif config.algo.name == "dcg_me_gecco":
                    metrics["num_evaluations"] = metrics["iteration"] * (config.batch_size + config.algo.actor_batch_size)
                elif config.algo.name == "memes":
                    metrics["num_evaluations"] = metrics["iteration"] * ((config.batch_size * config.algo.sample_number * config.algo.num_in_optimizer_steps) + config.batch_size)
                    #print(metrics["num_evaluations"])
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