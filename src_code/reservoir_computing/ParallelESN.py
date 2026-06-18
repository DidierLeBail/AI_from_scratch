"""Defines the Parallel Echo State Network described [here](https://arxiv.org/pdf/2601.22296).
"""

import torch
import torch.nn.functional as F

__all__ = [
    "ParallelESN"
]

def complexMatmul(input: torch.Tensor, other: torch.Tensor):
    """Same as `torch.matmul` but for complex-valued tensors.
    """
    real_part = torch.matmul(torch.real(input), torch.real(other)) - torch.matmul(torch.imag(input), torch.imag(other))
    imag_part = torch.matmul(torch.real(input), torch.imag(other)) + torch.matmul(torch.imag(input), torch.real(other))
    return real_part + 1j * imag_part

class MixingBlock(torch.nn.Module):
    """The mixing block of a parallel echo state network.

    A mixing block is a non-linear vector function applied to the hidden state of a given layer at a given time.
    Its number of trainable parameters is `kernelSize + 1`.

    Notes
    -----
    The input to a `MixingBlock` should be of size `B x 1 x n`, with `n > kernelSize`.
    Then the output is of size `B x 1 x n`.

    Parameters
    ----------
    kernelSize : int
        the number of complex components of `self.w_mix`

    Attributes
    ----------
    w_mix : torch.Tensor
        the 1d complex-valued tensor to convolve the input (size `1 x 1 x kernelSize`)
    b_mix : torch.Tensor
        the 0d complex-valued tensor, to add to the convolved input as a bias (size `1`)
    """

    def __init__(self, kernelSize: int, **kwargs):
        super().__init__()
        self.kernelSize = kernelSize

        self.w_mix = torch.ones( (1, 1, kernelSize), dtype=torch.complex32 )
        self.b_mix = torch.zeros(1, dtype=torch.complex32)

    def forward(self, h: torch.Tensor):
        return F.tanh( torch.real( torch.conv1d(h, self.w_mix, self.b_mix, padding="same") ) )

class ReservoirLayer(torch.nn.Module):
    """The reservoir layer of a parallel echo state network.
    
    It is used to compute the hidden state at time t and layer l,
    through a linear transformation of the hidden state at time t-1 and layer l and of the mixed state at time t and layer l-1.

    Notes
    -----
    The inputs `h` and `z` to a `ReservoirLayer` should be both of size `B x n_hid`.
    Then the output is of size `B x n_hid`.

    Parameters
    ----------
    n_hid : int
        the number of components of the hidden state
    leakyRate : float
        btw 0 and 1, `1 - leakyRate` is the proportion of the past hidden state contributing to the new hidden state
    
    Attributes
    ----------
    transMat : torch.Tensor
        the 1d complex-valued tensor of size `n_hid` serving as a diagonal transition matrix from the past to current hidden states
    inMat : torch.Tensor
        the 1d complex-valued tensor of size `n_hid` serving as the input weight matrix (transition from mixed to hidden states), that has a ring topology,
        which enables it to be stored as a 1d tensor
    bias : torch.Tensor
        the 1d complex-valued tensor of size `n_hid` serving as a bias for the transition from mixed to hidden states
    """

    def __init__(self, n_hid: int, leakyRate: float, **kwargs):
        super().__init__()
        self.n_hid = n_hid
        self.leakyRate = leakyRate

        self.transMat = torch.ones(n_hid, dtype=torch.complex32)
        self.inMat = torch.ones(n_hid, dtype=torch.complex32)
        self.bias = torch.zeros(n_hid, dtype=torch.complex32)
    
    def forward(self, h: torch.Tensor, z: torch.Tensor):
        return torch.mul(self.transMat, h) + self.leakyRate * (torch.mul(self.inMat, z) + self.bias)

class ReservoirFirstLayer(torch.nn.Module):
    """The first layer of the reservoir of a parallel echo state network.
    
    The difference with a deeper layer is that the matrix transition from mixed to hidden states (the input weight matrix) is dense of size
    `n_hid x n_in`.
    It is so because it maps the external input to the hidden dimension.

    Notes
    -----
    The inputs `h` and `x` to a `ReservoirFirstLayer` should be of sizes `B x n_hid` and `B x n_in x 1`.
    The output is of size `B x n_hid`.
    """

    def __init__(self, n_hid: int, n_in: int, leakyRate: float, **kwargs):
        super().__init__()
        self.n_hid = n_hid
        self.n_in = n_in
        self.leakyRate = leakyRate

        self.transMat = torch.ones(n_hid, dtype=torch.complex32)
        self.inMat = torch.ones( (n_hid, n_in), dtype=torch.complex32 )
        self.bias = torch.zeros(n_hid, dtype=torch.complex32)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor):
        return torch.mul(self.transMat, h) + self.leakyRate * (complexMatmul(self.inMat, x).squeeze(dim=-1) + self.bias)

class ParallelESN(torch.nn.Module):
    """The deep parallel echo state network.
    
    Parameters
    ----------
    n_layers : int
        the number of layers other than the first layer (nb of `ReservoirLayer` blocks)
    """

    def __init__(self,
        n_layers: int,
        n_hid: int,
        n_in: int,
        leakyRate: float
    ):
        super().__init__()
        self.n_layers = n_layers
        
    
    def forward(self, x: torch.Tensor):
        pass
