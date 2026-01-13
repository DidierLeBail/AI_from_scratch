"""Gathers datasets of images (integer-valued matrices),
for training vision-based diffusion models.
"""
from typing import Tuple
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms.functional import pad
import matplotlib.pyplot as plt

from configurations import Config_mnist

def single_dot_training_set(
        img_size: Tuple[int, int],
        set_size: int,
        seed: int,
        nb_channels: int,
        return_all: bool=False,
    ) -> torch.utils.data.TensorDataset:
        r"""Generate a set of black and white square matrices, where each matrix has a single white dot.
        All positions for the dot are equally likely.

        Parameters
        ----------
        img_size : Tuple[int, int]
            size of the produced matrices, so that the number of pixels
            in one image is img_size[0] * img_size[1]
        set_size : int
            number of produced matrices
        seed : int
            the seed for initializing the random generator used to build the dataset
        nb_channels : int
            the number of channels: the size of an image wil be
            `(nb_channels, *img_size)`
        return_all : bool, optional
            if `True`, ignore `set_size` and return all possible matrices
            with a single white dot (so img_size ** 2 matrices)
        """

        if return_all:
            res = -10 *torch.ones((img_size[0] * img_size[1], *img_size))
            for i in range(img_size[0]):
                for j in range(img_size[1]):
                    res[i * img_size[1] + j, i, j] = 10
            return TensorDataset(res)

        generator = torch.Generator(device='mps')
        generator.manual_seed(seed)

        # draw the dot position for every matrix
        pos = torch.stack(
            [
                torch.randint(
                    low=0,
                    high=high,
                    size=(set_size,),
                    generator=generator
                ) for high in img_size
            ]
        ).transpose(0, 1)

        # draw the images
        res = -10 * torch.ones((set_size, *img_size), dtype=int) / 2
        for k, (i, j) in enumerate(pos):
            res[k, i, j] = 10
        return TensorDataset(res)

class CustomMnist(MNIST):
    """Custom dataset with MNIST data.
    """

    def __init__(self,
        cfg: Config_mnist
    ):
        root = "src_code/diffusion_model/train_data/"
        MNIST.__init__(self,
                    root=root, train=True, download=True,
                    target_transform=None, transform=cfg.transform)

        self.data = self.data[:cfg.set_size]
        self.pad_size = (cfg.img_size[0] - 28) // 2
    
    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Parameters
        ----------
        index : int
            index
        """
        img = pad(self.data[index].to(dtype=torch.float32), [self.pad_size]) / 255

        if self.transform is not None:
            img = self.transform(img)

        return 2 * img - 1

def display_mat(
    mat: torch.Tensor
):
    r"""Displays an integer-valued matrix as an image.
    
    Use `plt.show()` to visualize the figure.
    
    Parameters
    ----------
    mat : torch.Tensor
        the matrix (2D tensor) to visualize
    """

    _, ax = plt.subplots(constrained_layout=True)
    plt.pcolormesh(
        mat,
        cmap='gray',
        edgecolors='gray',
        linewidth=1
    )
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
