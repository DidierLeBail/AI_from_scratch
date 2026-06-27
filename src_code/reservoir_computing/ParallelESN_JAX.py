"""JAX and FLAX implementation of the Parallel Echo State Network described
[here](https://arxiv.org/pdf/2601.22296).

For a jax and flax tuto, see:

https://docs.jaxstack.ai/en/latest/neural_net_basics.html

Also, note that dataloaders are not native in jax or flax, so preferentially use
the dataloaders of Tensorflow (more efficient that PyTorch e.g.).
"""

import jax, optax
from flax import nnx

try:
    devices = jax.devices('gpu')
except RuntimeError:
    devices = jax.devices('cpu')
device = devices[0]
print(device)

tab = jax.numpy.arange(3)
print(tab)
print(tab.devices())
