"""
Implement a simple diffusion model using backward Langevin dynamics and a score vector function learned by a neural network (differential model).
"""
from typing import List, Tuple, Callable, Union
import matplotlib.pyplot as plt
import torch

class Custom_rng:
    """Gathers methods to generate random objects.

    Allows easy control of the seed for repeatability.
    
    Parameters
    ----------
    seed : int
        seed for pseudo random generation
    kwargs : Dict
        passed to the constuctor of `torch.Generator`
    """
    def __init__(self,
        seed: int,
        **kwargs
    ):
        self.rng = torch.Generator(**kwargs)
        self.rng.manual_seed(seed)
    
    def single_dot_training_set(self,
        img_size: int,
        set_size: int,
        return_all: bool=False,
    ):
        r"""Generate a set of black and white square matrices, where each matrix has a single white dot.
        All positions for the dot are equally likely.

        Parameters
        ----------
        img_size : int
            size of the produced matrices, so that the number of pixels in one image is img_size ** 2
        set_size : int
            number of produced matrices
        return_all : bool, optional
            if `True`, ignore `set_size` and return all possible matrices with a single white dot (so img_size ** 2 matrices)
        seed : Union[int, None], optional
            the seed for the set generation
        """

        if return_all:
            res = torch.zeros((img_size ** 2, img_size, img_size), dtype=int)
            for i in range(img_size):
                for j in range(img_size):
                    res[i * img_size + j, i, j] = 1
            return res

        # draw the dot position for every matrix
        pos = torch.randint(low=0, high=img_size, size=(set_size, 2), generator=self.rng)
        
        # draw the images
        res = torch.zeros((set_size, img_size, img_size), dtype=int)
        for k, (i, j) in enumerate(pos):
            res[k, i, j] = 1
        return res

def display_mat(
    mat: torch.Tensor
):
    """Display an integer-valued matrix as an image.
    
    Parameters
    ----------
    mat : torch.Tensor
        the matrix to visualize
    """

    _, ax = plt.subplots(constrained_layout=True)
    plt.pcolormesh(
        mat,
        cmap='gnuplot2',
        edgecolors='gray',
        linewidth=1
    )
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    params = {
        "img_size": 2,
        "set_size": 3,
        "return_all": False
    }

    for param, val in params.items():
        print(f"{param}: {val}")
        print()
    
    rng = Custom_rng(seed=0)
    test_training_set = rng.single_dot_training_set(**params)

    for img in test_training_set:
        display_mat(img)
    plt.show()
