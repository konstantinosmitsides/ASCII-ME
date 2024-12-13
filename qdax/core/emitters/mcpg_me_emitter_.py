from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.mcpg_emitter_ import MCPGConfig, MCPGEmitter_0
from qdax.core.emitters.standard_emitters_ import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey
from dataclasses import dataclass

@dataclass
class MEMCPGConfig:
    """Configuration for PGAME Algorithm"""

    proportion_mutation_ga: float = 0.5
    no_agents: int = 512
    buffer_sample_batch_size: int = 2
    no_epochs: int = 16
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    discount_rate: float = 0.99
    greedy: float = 0.5
    cosine_similarity: bool = True
    std: float = 0.5
    
    
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
        
        if mcpg_no_agents == 0:
            ga_emitter = MixingEmitter(
                mutation_fn=None,
                variation_fn=variation_fn,
                variation_percentage=1.0,
                batch_size=ga_no_agents
        )
            super().__init__(emitters=(ga_emitter,))
            
        elif ga_no_agents == 0:
            mcpg_config = MCPGConfig(
                no_agents=mcpg_no_agents,
                buffer_sample_batch_size=config.buffer_sample_batch_size,
                buffer_add_batch_size=config.no_agents,
                no_epochs=config.no_epochs,
                learning_rate=config.learning_rate,
                discount_rate=config.discount_rate,
                clip_param=config.clip_param,
                std=config.std
            )

            mcpg_emitter = MCPGEmitter_0(
                config=mcpg_config, policy_net=policy_network, env=env
            )
        
            super().__init__(emitters=(mcpg_emitter,))
            
            # if config.greedy == 0.0:
            #     if config.cosine_similarity:

            #         mcpg_emitter = MCPGEmitter_0(
            #             config=mcpg_config, policy_net=policy_network, env=env
            #         )
                
            #         super().__init__(emitters=(mcpg_emitter,))
                
            #     else:
            #         mcpg_emitter = MCPGEmitter_0_not(
            #             config=mcpg_config, policy_net=policy_network, env=env
            #         )
                
            #         super().__init__(emitters=(mcpg_emitter,))
            
            # elif config.greedy == 0.5:

            #     mcpg_emitter = MCPGEmitter_05(
            #         config=mcpg_config, policy_net=policy_network, env=env
            #     )
            
            #     super().__init__(emitters=(mcpg_emitter,))
                
            # else:
            #     if config.cosine_similarity:
                    
            #         mcpg_emitter = MCPGEmitter_1(
            #             config=mcpg_config, policy_net=policy_network, env=env
            #         )
                
            #         super().__init__(emitters=(mcpg_emitter,))
            #     else:
                        
            #             mcpg_emitter = MCPGEmitter_1_not(
            #                 config=mcpg_config, policy_net=policy_network, env=env
            #             )
                    
            #             super().__init__(emitters=(mcpg_emitter,))
                
            
        else:
            
            mcpg_config = MCPGConfig(
                no_agents=mcpg_no_agents,
                buffer_sample_batch_size=config.buffer_sample_batch_size,
                buffer_add_batch_size=config.no_agents,
                no_epochs=config.no_epochs,
                learning_rate=config.learning_rate,
                discount_rate=config.discount_rate,
                clip_param=config.clip_param,
                std=config.std,
            )
            
            ga_emitter = MixingEmitter(
                mutation_fn=None,
                variation_fn=variation_fn,
                variation_percentage=1.0,
                batch_size=ga_no_agents
            )
            

            mcpg_emitter = MCPGEmitter_0(
                config=mcpg_config, policy_net=policy_network, env=env
            )


            super().__init__(emitters=(mcpg_emitter, ga_emitter))


            # if config.greedy == 0.0:
            #     if config.cosine_similarity:
                    
            #         if config.clip_param == 0.2:

            #             mcpg_emitter = MCPGEmitter_0(
            #                 config=mcpg_config, policy_net=policy_network, env=env
            #             )
            #         else:
            #             mcpg_emitter = MCPGEmitter_0_exps(
            #                 config=mcpg_config, policy_net=policy_network, env=env
            #             )
                        
                
            #         super().__init__(emitters=(mcpg_emitter, ga_emitter))
            #     else:
            #         mcpg_emitter = MCPGEmitter_0_not(
            #             config=mcpg_config, policy_net=policy_network, env=env
            #         )
                
            #         super().__init__(emitters=(mcpg_emitter, ga_emitter))

            
            # elif config.greedy == 0.5:

            #     mcpg_emitter = MCPGEmitter_05(
            #         config=mcpg_config, policy_net=policy_network, env=env
            #     )
            
            #     super().__init__(emitters=(mcpg_emitter, ga_emitter))
                
            # else:
            #     if config.cosine_similarity:
                    
            #         mcpg_emitter = MCPGEmitter_1(
            #             config=mcpg_config, policy_net=policy_network, env=env
            #         )
                
            #         super().__init__(emitters=(mcpg_emitter, ga_emitter))
                    
            #     else:
                        
            #             mcpg_emitter = MCPGEmitter_1_not(
            #                 config=mcpg_config, policy_net=policy_network, env=env
            #             )
                    
            #             super().__init__(emitters=(mcpg_emitter, ga_emitter))
            
