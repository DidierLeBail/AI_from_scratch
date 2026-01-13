"""Contains all configs data, that are used by `main.py`.
"""

class Config_ddpm:
    n_forward_steps = 1000
    img_size = 32
    n_backward_steps = 200
    backward_strategy = "ddim"
    b_start = 1e-4
    b_end = 0.02

class Config_mnist:
    img_size = (32, 32)
    set_size = 8
    transform = None

class Config_training:
    model_path = "src_code/diffusion_model/models_saved/mnist.pt"
    n_epochs = 3000
    lr = 1e-4
    batch_size = min(64, Config_mnist.set_size)
