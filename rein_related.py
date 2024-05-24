from dataclasses import dataclass
from typing import Any, Union, Tuple, Callable, Optional
from functools import partial
import time

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import optax

from qdax import environments

from IPython.display import HTML
from brax.io import html


