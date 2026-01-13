"""Contains all configs data, that are used by `main.py`.
"""

class Config_dfp:
    n_forward_steps = 1000
    b_start = 1e-4
    b_end = 0.02

class Config_drp:
    sampling_strategy = "ddim"
    n_backward_steps = min(200, Config_dfp.n_forward_steps)
    b_start = Config_dfp.b_start
    b_end = Config_dfp.b_end

class Config_mnist:
    img_size = (32, 32)
    set_size = 8
    transform = None

class Config_training:
    model_path = "src_code/diffusion_model/models_saved/mnist.pt"
    n_epochs = 3000
    lr = 1e-4
    batch_size = min(64, Config_mnist.set_size)
