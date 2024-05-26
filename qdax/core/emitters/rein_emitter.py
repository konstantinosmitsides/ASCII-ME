from dataclasses import dataclass
from functools import partial
from math import floor 
from typing import Callable, Tuple, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from qdax.environments.base_wrappers import QDEnv
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from rein_related import *

from qdax.core.emitters.es_novelty_archives import (
    EmptyNoveltyArchive,
    NoveltyArchive,
    ParallelNoveltyArchive,
    RepertoireNoveltyArchive,
    SequentialNoveltyArchive,
    SequentialScanNoveltyArchive,
)
from qdax.core.emitters.emitter import Emitter, EmitterState

@dataclass
class REINaiveConfig:
    """Configuration for the REINaive emitter.
    
    Args:
        rollout_number: num of rollouts for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate  (IS THIS PARAMETER SPACE EXPLORATION?)
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation
        
        num_generations_sample: frequency of archive-sampling
        
        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate: obvious
        l2_coefficient: coefficient for regularisation
        
        novelty_nearest_neighbors: num of nearest neigbors for novelty computation
        use_novelty_archive: if True, use novelty archive for novelty (default is to use the content of the reperoire)
        use_novelty_fifo: if True, use fifo archive for novelty (default is to use the content of the repertoire)
        fifo_size: size of the novelty fifo bugger if used
        
        proprtion_explore: proportion of explore
    """
    batch_size: int = 32
    num_rein_training_steps: int = 10
    buffer_size: int = 100000
    rollout_number: int = 10
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True
    
    num_generations_sample: int = 10
    
    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = False
    
    novelty_nearest_neighbors: int = 10
    use_novelty_archive: bool = False
    use_novelty_fifo: bool = False
    fifo_size: int = 100000
    
    proportion_explore: float = 0.0


class REINaiveEmitterState(EmitterState):
    """Contains replay buffer.
    """
    trajectory_buffer: TrajectoryBuffer
    random_key: RNGKey
    
    
class REINaiveEmitter(Emitter):
    """
    An emitter that uses gradients approximated through rollouts.
    It dedicates part of the process on REINFORCE for fitness gradients and part
    to exploration gradients.
    
    This scan version scans through parents isntead of performing all REINFORCE
    operations in parallel, to avoid memory overload issue.
    """
    
    def __init__(
        self,
        config: REINaiveConfig,
        policy_network: nn.Module,
        env: QDEnv,
    ) -> None:
        self._config = config
        self._policy = policy_network
        self._env = env
            
            
        # SET UP THE LOSSES
        
        # Init optimizers
        
        self._policies_optmizer = optax.adam(
            learning_rate=self._config.learning_rate
            )
        
        @property
        def batch_size(self) -> int:
            """
            Returns:
                int: the batch size emitted by the emitter.
            """
            return self._config.batch_size
        
        @property 
        def use_all_data(self) -> bool:
            """Whether to use all data or not when used along other emitters.
            """
            return True
        
        @partial(jax.jit, static_argnames=("self",))
        def init(
            self,
            init_genotypes: Genotype,
            random_key: RNGKey,
        ) -> Tuple[REINaiveEmitter, RNGKey]:
            """Initializes the emitter.

            Args:
                init_genotypes: The initial population.
                random_key: A random key.

            Returns:
                The initial state of the REINAiveEmitter, a new random key.
            """

            observation_size = self._env.observation_size
            action_size = self._env.action_size
            descriptor_size = self._env.state_descriptor_length
            
            # Init trajectory buffer
            dummy_transition = QDTransition.init_dummy(
                observation_size=observation_size,
                action_size=action_size,
                descriptor_dim=descriptor_size,
            )
            
            trajectory_buffer = TrajectoryBuffer.init(
                buffer_size=self._config.buffer_size,
                transition=dummy_transition,
                env_batch_size=self._config.batch_size,
                episode_length=self._env.episode_length,
            )
            
            random_key, subkey = jax.random.split(random_key)
            emitter_state = REINaiveEmitterState(
                trajectory_buffer=trajectory_buffer,
                random_key=subkey,
            )
            
            return emitter_state, random_key
        
        @partial(jax.jit, static_argnames=("self",))
        def emit(
            self,
            repertoire: Repertoire,
            emitter_state: REINaiveEmitterState,
            random_key: RNGKey,
        ) -> Tuple[Genotype, RNGKey]:
            """Do a step of REINFORCE emission.

            Args:
                repertoire: the current repertoire of genotypes.
                emitter_state: the state of the emitter used
                random_key: random key

            Returns:
                A batch of offspring, the new emitter state and a new key.
            """
            
            batch_size = self._config.batch_size
            
            # sample parents
            parents, random_key = repertoire.sample(
                batch_size=batch_size,
                random_key=random_key,
            )
            
            offsprings_rein = self.emit_rein(emitter_state, parents)
            
            genotypes = offsprings_rein
            
            return genotypes, {}, random_key
        
        @partial(jax.jit, static_argnames=("self",))
        def emit_rein(
            self,
            emitter_state: REINaiveEmitterState,
            parents: Genotype,
        ) -> Genotype:
            """Emit the offsprings generated through REINFORCE mutation.

            Args:
                emitter_state: the state of the emitter used, contains
                the trahectory buffer.
                parents: the parents selected to be applied gradients 
                to mutate towards better performance.

            Returns:
                A new set of offspring.
            """
            
            # Do a step of REINFORCE emission
            mutation_fn = partial(
                self._mutation_function_rein,
                emitter_state=emitter_state,
            )
            offsprings = jax.vmap(mutation_fn)(parents)
            
            return offsprings
        
        @partial(jax.jit, static_argnames=("self",))
        def _mutation_function_rein(
            self,
            policy_params: Genotype,
            emitter_state: REINaiveEmitterState,
        ) -> Genotype:
            """Apply REINFORCE mutation to a policy via multiple steps of gradient descent.

            Args:
                policy_params: a policy, supposed to be a differentiable neuaral network.
                emitter_state: the current state of the emitter, containing among others,
                the trajectory buffer.

            Returns:
                The updated parameters of the neural network.
            """
            
            # Define new policy optimizer state
            policy_optimizer_state = self._policies_optmizer.init(policy_params)
            
            def scan_train_policy(
                carry: Tuple[REINaiveEmitterState, Genotype, optax.OptState],
                unused: Any,
            ) -> Tuple[Tuple[REINaiveEmitterState, Genotype, optax.OptState], Any]:
                """Scans through the parents and applies REINFORCE training.
                """
                
                emitter_state, parent, policy_optimizer_state = carry
                
                (
                    new_emitter_state,
                    new_policy_params,
                    new_policy_optimizer_state,
                ) = self._train_policy_(
                    emitter_state,
                    policy_params,
                    policy_optimizer_state,
                )
                return (
                    new_emitter_state,
                    new_policy_params,
                    new_policy_optimizer_state,
                ), ()
                
                (emitter_state, policy_params, policy_optimizer_state,), _ = jax.lax.scan(
                    scan_train_policy,
                    (emitter_state, policy_params, policy_optimizer_state),
                    (),
                    length=self._config.num_rein_training_steps,
                )
                
                return policy_params
            
        @partial(jax.jit, static_argnames=("self",))
        def _train_policy_(
            self,
            emitter_state: REINaiveEmitterState,
            policy_params: Genotype,
            policy_optimizer_state: optax.OptState,
        ) -> Tuple[REINaiveEmitterState, Genotype, optax.OptState]:
            """Train the policy network with REINFORCE.

            Args:
                emitter_state: the current state of the emitter.
                policy_params: the current parameters of the policy network.
                policy_optimizer_state: the current state of the optimizer.

            Returns:
                The updated state of the emitter, the updated policy parameters
                and the updated optimizer state.
            """
            
            trajectory_buffer = emitter_state.trajectory_buffer
            
            # Sample trajectories
            transitions = trajectory_buffer.sample(
                batch_size=self._config.batch_size,
                random_key=emitter_state.random_key,
            )
            
            # Compute the loss and the gradients
            loss, grads = self.loss_reinforce(
                policy_params,
                transitions.observation,
                transitions.action,
                transitions.logp,
                transitions.mask,
                transitions.return_standardized,
            )
            
            # Apply gradients
            updates, new_policy_optimizer_state = self._policies_optmizer.update(
                grads,
                policy_optimizer_state,
            )
            
            new_policy_params = optax.apply_updates(policy_params, updates)
            
            # Update the emitter state
            new_emitter_state = REINaiveEmitterState(
                trajectory_buffer=trajectory_buffer,
                random_key=emitter_state.random_key,
            )
            
            return new_emitter_state, new_policy_params, new_policy_optimizer_state
        