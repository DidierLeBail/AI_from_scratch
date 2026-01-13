"""Contains all configs data, that are used by `main.py`.
"""

class Config_dfp:
    r"""Config for a diffusion forward process.

    Parameters
    ----------
    n_forward_steps : int
        nb of noising steps
    b_start : float
        initial value of :math:`\beta`
    b_end : float
        final value of :math:`\beta`
    """
    n_forward_steps = 1000
    b_start = 1e-4
    b_end = 0.02

class Config_drp:
    r"""Config for a diffusion reverse process.

    Parameters
    ----------
    sampling_strategy: Literal['ddim', 'ddpm']
        the denoising strategy, either a deterministic ('ddim')
        or stochastic ('ddpm') function of the seed
    n_backward_steps : int
        nb of denoising steps
    b_start : float
        initial value of :math:`\beta`
    b_end : float
        final value of :math:`\beta`
    """
    sampling_strategy = "ddim"
    n_backward_steps = min(200, Config_dfp.n_forward_steps)
    b_start = Config_dfp.b_start
    b_end = Config_dfp.b_end

class Config_mnist:
    r"""Config for building a MNIST dataset.

    Parameters
    ----------
    img_size: Tuple[int, int]
        desired size of the samples.
        The initial size is (28, 28) but if larger, the original samples are padded.
        Note that (for now), only square images are handled.
    set_size : int
        nb of training samples: allow a subset of the original MNIST to be considered
    transform : Union[Callable, None]
        transformation eventually applied to the samples, for data augmentation
    """
    img_size = (32, 32)
    set_size = 8
    transform = None

class Config_training:
    r"""Config for training the denoiser of a diffusion model.

    Parameters
    ----------
    model_path: str
        where to save the trained model
    n_epochs : int
        nb of training epochs ;
        in one epoch, the model has seen exactly once every training example
    lr : float
        the learning rate, kept constant here.
        If the learning rate is too large,
        the test error can suddenly jump during training, and not decrease after that.
    batch_size : int
        the number of samples on which the error made by the model is averaged:
        the larger the batch size, the more precise the evaluation of the gradient ;
        but the smaller the number of updates per epoch.
        Also a large batch size requires more (gpu) RAM.
    """
    model_path = "src_code/diffusion_model/models_saved/mnist.pt"
    n_epochs = 3000
    lr = 1e-4
    batch_size = min(64, Config_mnist.set_size)
