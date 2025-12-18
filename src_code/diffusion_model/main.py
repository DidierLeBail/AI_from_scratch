"""
Implement a simple diffusion model using backward Langevin dynamics and a score vector function learned by a neural network (differential model).
"""
from typing import List, Tuple, Callable, Union
import tensorflow as tf

def generate_single_dot_training_set(
    img_size: int,
    set_size: int,
    return_all: bool=False,
    seed: Union[int, None]=None
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
        res = tf.zeros((img_size ** 2, img_size, img_size), dtype=int)
        for i in range(img_size):
            for j in range(img_size):
                res[i * img_size + j, i, j] = 1
        return res

    # draw the dot position for every matrix
    rng = tf.random.Generator.from_seed(seed)
    pos = rng.uniform(set_size, minval=0, maxval=img_size, dtype=int)
    exit()
    
    # draw the images
    res = tf.zeros((set_size, img_size, img_size), dtype=int)
    for i, j in pos:
        res[i * img_size + j, i, j] = 1
    return res

if __name__ == "__main__":
    set_size = 3
    img_size = 2
    test_training_set = generate_single_dot_training_set(img_size, set_size, return_all=False, seed=0)
    for img in test_training_set:
        print(img)
        print()
