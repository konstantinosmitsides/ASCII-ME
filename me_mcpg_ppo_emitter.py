from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.mcpg_emitter import MCPGConfig, MCPGEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey
from dataclasses import dataclass
from pure_ppo_emitter import PurePPOEmitter, PurePPOConfig

class MEMCPGPPOConfig:
    """Configuration for PGAME Algorithm"""

    
    proportion_mutation_ga: float = 0.5
    no_agents: int = 511
    buffer_sample_batch_size: int = 2
    buffer_add_batch_size: int = 512
    no_epochs: int = 16
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    LR: float = 1e-3
    NUM_ENVS: int = 256 #2048
    NUM_STEPS: int = 80 #10
    TOTAL_TIMESTEPS: int = 5e7
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENTROPY_COEFF: float = 0.0
    VF_COEFF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ANNEAL_LR: bool = False
    NORMALIZE_ENV: bool = True
    NO_ADD: int = 1
    GREEDY_AGENTS: int = 1


class MEMCPGPPOEmitter(MultiEmitter):
    def __init__(
        self,
        config: MEMCPGPPOConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:
        
        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn
        
        ga_no_agents = int(self._config.proportion_mutation_ga * config.no_agents)
        mcpg_no_agents = config.no_agents - ga_no_agents
        
        mcpg_config = MCPGConfig(
            no_agents=mcpg_no_agents,
            buffer_sample_batch_size=config.buffer_sample_batch_size,
            buffer_add_batch_size=config.buffer_add_batch_size,
            #batch_size=config.batch_size,
            no_epochs=config.no_epochs,
            learning_rate=config.learning_rate,
            clip_param=config.clip_param
        )
        
        ppo_config = PurePPOConfig(
            LR=config.LR,
            NUM_ENVS=config.NUM_ENVS,
            NUM_STEPS=config.NUM_STEPS,
            TOTAL_TIMESTEPS=config.TOTAL_TIMESTEPS,
            GAMMA=config.GAMMA,
            GAE_LAMBDA=config.GAE_LAMBDA,
            CLIP_EPS=config.CLIP_EPS,
            ENTROPY_COEFF=config.ENTROPY_COEFF,
            VF_COEFF=config.VF_COEFF,
            MAX_GRAD_NORM=config.MAX_GRAD_NORM,
            ANNEAL_LR=config.ANNEAL_LR,
            NORMALIZE_ENV=config.NORMALIZE_ENV,
            NO_ADD=config.NO_ADD,
            GREEDY_AGENTS=config.GREEDY_AGENTS
        )
        
        
        mcpg_emitter = MCPGEmitter(
            config=mcpg_config, 
            policy_net=policy_network, 
            env=env,
        )
        
        ppo_emitter = PurePPOEmitter(
            config=ppo_config,
            policy_net=policy_network,
            env=env,
        )
        
        ga_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_no_agents
        )
        
        super().__init__(emitters=(mcpg_emitter, ppo_emitter, ga_emitter))
        