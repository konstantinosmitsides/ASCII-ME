from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.mcpg_emitter_trans import MCPGConfig, MCPGEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey
from dataclasses import dataclass

@dataclass
class MEMCPGConfig:
    """Configuration for PGAME Algorithm"""

    proportion_mutation_ga: float = 0.5
    no_agents: int = 256
    buffer_sample_batch_size: int = 32
    grad_steps: int = 16
    buffer_size: int = 512000
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    clip_param: float = 0.2
    
    
class MEMCPGEmitter(MultiEmitter):
    def __init__(
        self,
        config: MEMCPGConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn
        
        #print(config.batch_size)

        ga_no_agents = int(self._config.proportion_mutation_ga * config.no_agents)
        mcpg_no_agents = config.no_agents - ga_no_agents
        
        mcpg_config = MCPGConfig(
            no_agents=mcpg_no_agents,
            buffer_sample_batch_size=config.buffer_sample_batch_size,
            grad_steps=config.grad_steps,
            buffer_size=config.buffer_size,
            learning_rate=config.learning_rate,
            max_grad_norm=config.max_grad_norm,
            clip_param=config.clip_param
        )

        # define the quality emitter
        mcpg_emitter = MCPGEmitter(
            config=mcpg_config, policy_net=policy_network, env=env
        )
        
        ga_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_no_agents
        )
        
        super().__init__(emitters=(mcpg_emitter, ga_emitter))