""" Implements neural networks models that are commonly found in the RL literature."""

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


class MLPDC(nn.Module):
    """Descriptor-conditioned MLP module."""

    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None

    @nn.compact
    def __call__(self, obs: jnp.ndarray, desc: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, desc], axis=-1)
        for i, hidden_size in enumerate(self.layer_sizes):

            if i != len(self.layer_sizes) - 1:
                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=self.kernel_init,
                    use_bias=self.bias,
                )(hidden)
                hidden = self.activation(hidden)  # type: ignore

            else:
                if self.kernel_init_final is not None:
                    kernel_init = self.kernel_init_final
                else:
                    kernel_init = self.kernel_init

                hidden = nn.Dense(
                    hidden_size,
                    kernel_init=kernel_init,
                    use_bias=self.bias,
                )(hidden)

                if self.final_activation is not None:
                    hidden = self.final_activation(hidden)

        return hidden


class QModule(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLP(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden)
            res.append(q)
        return jnp.concatenate(res, axis=-1)


class QModuleDC(nn.Module):
    """Q Module."""

    hidden_layer_sizes: Tuple[int, ...]
    n_critics: int = 2

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, actions: jnp.ndarray, desc: jnp.ndarray
    ) -> jnp.ndarray:
        hidden = jnp.concatenate([obs, actions], axis=-1)
        res = []
        for _ in range(self.n_critics):
            q = MLPDC(
                layer_sizes=self.hidden_layer_sizes + (1,),
                activation=nn.relu,
                kernel_init=jax.nn.initializers.lecun_uniform(),
            )(hidden, desc)
            res.append(q)
        return jnp.concatenate(res, axis=-1)

'''
_half_log2pi = 0.5 * jnp.log(2 * jnp.pi)
EPS = 1e-8

class MLPRein(nn.Module):
    """MLP-REINFORCE module."""

    action_size: int
    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = None
    
    def setup(self):
        # Define the layers
        self.hidden_layers = [nn.Dense(size, kernel_init=self.kernel_init) for size in self.layer_sizes]
        self.mean = nn.Dense(self.action_size, kernel_init=self.kernel_init_final)
        #self.log_std = self.param('log_std', self.kernel_init_final or self.kernel_init, (self.action_size,), init_fn=jax.nn.initializers.zeros)
            
    def distribution_params(self, obs: jnp.ndarray):
        hidden = obs
        for hidden_layer in self.hidden_layers:
            hidden = self.activation(hidden_layer(hidden))

        mean = self.mean(hidden)
        log_std = self.log_std
        std = jnp.exp(log_std)

        return mean, log_std, std

    def logp(self, params, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # Distribution parameters
        mean, log_std, std = self.apply(params, obs, method=self.distribution_params)

        # Log probability
        logp = jnp.sum(-0.5 * jnp.square((action - mean)/(std + EPS)) - _half_log2pi - log_std, axis=-1)

        return logp

    def entropy(self, params, obs: jnp.ndarray) -> jnp.ndarray:
        # Distribution parameters
        _, _, std = self.apply(params, obs, method=self.distribution_params)

        entropy = self.action_size * (0.5 + _half_log2pi) + 0.5 * jnp.log(jnp.prod(std))
        return entropy

    #def __call__(self, params, random_key, obs: jnp.ndarray):
    #    return self.sample(params, random_key, obs)

    def sample(self, params, random_key, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Distribution parameters
        mean, log_std, std = self.apply(params, obs, method=self.distribution_params)

        # Sample action
        rnd = jax.random.normal(random_key, shape=mean.shape)
        action = mean + rnd * std

        # Log probability
        logp = jnp.sum(-0.5 * jnp.square((action - mean)/(std + EPS)) - _half_log2pi - log_std, axis=-1)

        return action, logp
        
'''
'''
_half_log2pi = 0.5 * jnp.log(2 * jnp.pi)
EPS = 1e-8

class MLPRein(nn.Module):
    """MLP-REINFORCE module."""

    action_size: int
    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = jax.nn.initializers.orthogonal(scale=0.01)
    
    def setup(self):
        # Define the layers
        self.hidden_layers = [nn.Dense(size, kernel_init=self.kernel_init) for size in self.layer_sizes]
        self.mean = nn.Dense(self.action_size, kernel_init=self.kernel_init_final)
        self.log_std = self.param("log_std", lambda _, shape: -1.*jnp.ones(shape), (self.action_size,))
        
    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        hidden = obs
        for hidden_layer in self.hidden_layers:
            hidden = self.activation(hidden_layer(hidden))

        mean = self.mean(hidden)
        log_std = self.log_std
        std = jnp.exp(log_std)

        return mean, log_std, std

    def distribution_params(self, obs: jnp.ndarray):
        mean, log_std, std = self.__call__(obs)
        return mean, log_std, std

    def logp(self, params, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # Distribution parameters
        mean, log_std, std = self.apply({'params': params}, obs, method=self.distribution_params)

        # Log probability
        logp = jnp.sum(-0.5 * jnp.square((action - mean)/(std + EPS)) - _half_log2pi - log_std, axis=-1)

        return logp

    def entropy(self, params, obs: jnp.ndarray) -> jnp.ndarray:
        # Distribution parameters
        _, _, std = self.apply({'params': params}, obs, method=self.distribution_params)

        entropy = self.action_size * (0.5 + _half_log2pi) + 0.5 * jnp.log(jnp.prod(std))
        return entropy

    def sample(self, params, random_key, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Distribution parameters
        mean, log_std, std = self.apply({'params': params}, obs, method=self.distribution_params)

        # Sample action
        rnd = jax.random.normal(random_key, shape=mean.shape)
        action = mean + rnd * std

        # Log probability
        logp = jnp.sum(-0.5 * jnp.square((action - mean)/(std + EPS)) - _half_log2pi - log_std, axis=-1)

        return action, logp
'''



_half_log2pi = 0.5 * jnp.log(2 * jnp.pi)
EPS = 1e-8

#def debug_trace(variable, name="Variable"):
#    print(f"{name}: type={type(variable)}, shape={getattr(variable, 'shape', 'N/A')}, is_jax_array={isinstance(variable, jnp.ndarray)}")

class MLPRein(nn.Module):
    """MLP-REINFORCE module."""

    action_size: int
    layer_sizes: Tuple[int, ...]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_uniform()
    final_activation: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
    bias: bool = True
    kernel_init_final: Optional[Callable[..., Any]] = jax.nn.initializers.orthogonal(scale=0.01) # this is specific for ant_uni
    
    
    def setup(self):
        # this is specific for ant_uni
        self.hidden_layers = [nn.Dense(size, kernel_init=self.kernel_init, use_bias=self.bias) for size in self.layer_sizes]
        self.mean = nn.Dense(self.action_size, kernel_init=self.kernel_init, use_bias=self.bias) #kernel_init=self.kernel_init_final)
        self.log_std = self.param("log_std", lambda _, shape: -1.0 * jnp.ones(shape), (self.action_size,))

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        hidden = obs
        for hidden_layer in self.hidden_layers:
            hidden = self.activation(hidden_layer(hidden))



        mean = self.mean(hidden)
        
        return mean
        
        
        #return self.sample(random_key, obs)[0]

    def distribution_params(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        hidden = obs
        for hidden_layer in self.hidden_layers:
            hidden = self.activation(hidden_layer(hidden))



        mean = self.mean(hidden)
        log_std = self.log_std
        std = jnp.exp(log_std)

        return mean, log_std, std, hidden
    
    def logp(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        #mean, log_std, std, _ = self.apply(params, obs, method=self.distribution_params)
        mean, log_std, std, _ = self.distribution_params(obs)
        #debug_trace(mean, "mean in logp")
        #debug_trace(log_std, "log_std in logp")
        #debug_trace(std, "std in logp")

        logp = jnp.sum(-0.5 * jnp.square((action - mean) / (std + EPS)) - _half_log2pi - log_std, axis=-1)
        return logp

    def entropy(self, params, obs: jnp.ndarray) -> jnp.ndarray:
        _, _, std, _ = self.apply(params, obs, method=self.distribution_params)
        entropy = self.action_size * (0.5 + _half_log2pi) + 0.5 * jnp.log(jnp.prod(std))
        return entropy

    def sample(self, random_key, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        #mean, log_std, std, _ = self.apply(params, obs, method=self.distribution_params)
        mean, log_std, std, _ = self.distribution_params(obs)

        rnd = jax.random.normal(random_key, shape=mean.shape)
        action = jax.lax.stop_gradient(mean + rnd * std)
        logp = jnp.sum(-0.5 * jnp.square((action - mean) / (std + EPS)) - _half_log2pi - log_std, axis=-1)
        return action, logp
