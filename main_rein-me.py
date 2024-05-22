import os
import logging
import time
from dataclasses import dataclass
from functools import partial
from math import floor
from typing import Any, Dict, Tuple, List, Callable

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.map_elites import MAPElites
from qdax.types import RNGKey, Genotype
from qdax.utils.sampling import sampling 
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids, MapElitesRepertoire
from qdax.core.neuroevolution.networks.networks import MLP

from set_up_brax import (
    get_behavior_descriptor_length_brax,
    get_environment_brax,
    get_policy_struc_brax,
    get_reward_offset_brax,
    get_scoring_function_brax,
)



# HERE IMPORT YOUR EMITTER AND EMITTER CONFIG 


def set_up_envs(
    config: ConfigStore,
    batch_size: int,
    random_key: RNGKey,
) -> Tuple[Any, Callable, Any, Genotype, float, int, int, RNGKey]:
    
    # Init environment and population of controllers
    print("Env name: ", config.env_name)
    
    # Initializing environment 
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
    
        
    