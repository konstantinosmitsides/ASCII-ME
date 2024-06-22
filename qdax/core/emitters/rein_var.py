#from dataclasses import dataclass
from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.rein_emitter_advanced import REINaiveConfig, REINaiveEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey
from dataclasses import dataclass

@dataclass
class REINConfig:
    """Configuration for PGAME Algorithm"""

    proportion_mutation_ga: float = 0.5
    batch_size: int = 32
    num_rein_training_steps: int = 10
    buffer_size: int = 320000
    rollout_number: int = 100
    discount_rate: float = 0.99
    adam_optimizer: bool = True
    learning_rate: float = 1e-3
    temperature: float = 0.


class REINEmitter(MultiEmitter):
    def __init__(
        self,
        config: REINConfig,
        policy_network: nn.Module,
        env: QDEnv,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._policy_network = policy_network
        self._env = env
        self._variation_fn = variation_fn
        
        print(config.batch_size)

        ga_batch_size = int(self._config.proportion_mutation_ga * config.batch_size)
        rein_batch_size = config.batch_size - ga_batch_size

        rein_config = REINaiveConfig(
            batch_size=rein_batch_size,
            num_rein_training_steps=config.num_rein_training_steps,
            buffer_size=config.buffer_size,
            rollout_number=config.rollout_number,
            discount_rate=config.discount_rate,
            adam_optimizer=config.adam_optimizer,
            learning_rate=config.learning_rate,
            temperature=config.temperature
        )

        # define the quality emitter
        rein_emitter = REINaiveEmitter(
            config=rein_config, policy_network=policy_network, env=env
        )

        # define the GA emitter
        ga_emitter = MixingEmitter(
            mutation_fn = None,
            #mutation_fn=lambda x, r: (x, r),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=ga_batch_size,
        )

        super().__init__(emitters=(rein_emitter, ga_emitter))
