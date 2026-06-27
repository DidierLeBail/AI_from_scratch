"""Reproduce the paper
[ParalESN: Enabling parallel information processing in Reservoir Computing]
(https://arxiv.org/pdf/2601.22296).

Notes
-----
To have an efficient implementation of a ParalESN, the parallel associative scan
for matrix linear combination must be implemented on gpu with autodifferentiation.
This requires either doing it by ourselves (using Triton) or migrating to JAX.
However, jax is not compatible with Apple M4 gpu, so this option requires
using NVIDIA (e.g.) gpu.

Then, if no compatible gpu is detected, we will switch to cpu (
but then the performance may degrade significantly).
"""

from ParallelESN import ParallelESN

