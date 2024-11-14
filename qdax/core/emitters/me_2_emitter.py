from typing import Callable, Tuple

import flax.linen as nn

from qdax.core.emitters.multi_emitter import MultiEmitter
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.environments.base_wrappers import QDEnv
from qdax.types import Params, RNGKey
from dataclasses import dataclass

@dataclass
class ME2CONFIG:
    """Configuration for PGAME Algorithm"""

    proportion_mutation_ga: float = 0.5
    batch_size: int = 512    
    
class ME2Emitter(MultiEmitter):
    def __init__(
        self,
        config: ME2CONFIG,
        variation_fn: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
        variation_fn_2: Callable[[Params, Params, RNGKey], Tuple[Params, RNGKey]],
    ) -> None:

        self._config = config
        self._variation_fn = variation_fn
        self._variation_fn_2 = variation_fn_2
        
        #print(config.batch_size)
        
        batch_size = int(config.batch_size * config.proportion_mutation_ga)
        batch_size_2 = config.batch_size - batch_size
        
        ga_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size
    )
        ga_emitter_2 = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn_2,
            variation_percentage=1.0,
            batch_size=batch_size_2,
        )
    
        
        super().__init__(emitters=(ga_emitter, ga_emitter_2))
        

        
