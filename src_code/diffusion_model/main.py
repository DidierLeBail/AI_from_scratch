"""
Implement a simple diffusion model using backward Langevin dynamics and a score vector function learned by a neural network (differential model).
"""

from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from platform import system

from configurations import Config_dfp, Config_mnist, Config_training, Config_drp
import models as models
import datasets as datasets

class DiffusionForwardProcess:
    r"""Forward diffusion process, to add a Gaussian noise to an image.
     
    References
    ----------
    We implement the algorithm described in [1]_.

    .. [1] Ho, J., Jain, A., & Abbeel, P. (2020).
       Denoising diffusion probabilistic models.
       Advances in neural information processing systems, 33, 6840-6851.
    """
    
    def __init__(self,
        cfg: Config_dfp
    ):
        self.n_steps = cfg.n_forward_steps

        # Precomputing alpha_bar for all t's.
        a_bar = torch.cumprod(
            1 - torch.linspace(cfg.b_start, cfg.b_end, self.n_steps),
            dim=0
        )
        self.sqrt_a_bar = torch.sqrt(a_bar)
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)
        
    def add_noise(self,
        original: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ):
        r"""Adds noise to a batch of original images at time step `t`.
        
        Parameters
        ----------
        original : torch.Tensor
            the input, of shape B x C x H x W
        noise : torch.Tensor
            the noise to add to the input, of shape B x C x H x W
        t : torch.Tensor[int]
            the time step *k* is associated to the *k* th image in the batch (shape B)
        """
        return self.sqrt_a_bar.to(original.device)[t][:, None, None] * original \
            + self.sqrt_one_minus_a_bar.to(original.device)[t][:, None, None] * noise

class DiffusionReverseProcess:
    r"""Reverse diffusion process, to denoise an image.
     
    References
    ----------
    We implement the algorithms described in [1]_ and [2]_.

    .. [1] Ho, J., Jain, A., & Abbeel, P. (2020).
       Denoising diffusion probabilistic models.
       Advances in neural information processing systems, 33, 6840-6851.
    .. [2] Song, J., Meng, C., & Ermon, S. (2020).
       Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502.
    """
    
    def __init__(self,
        cfg: Config_drp
    ):
        self.n_backward_steps = cfg.n_backward_steps

        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(cfg.b_start, cfg.b_end, cfg.n_backward_steps)
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.sqrt_a_bar = torch.sqrt(self.a_bar)
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - self.a_bar)

        # the sampling strategy
        if cfg.sampling_strategy == "ddim":
            self.sample_prev_time = self.sample_prev_time_ddim
        elif cfg.sampling_strategy == "ddim":
            self.sample_prev_time = self.sample_prev_time_ddpm
        else:
            raise ValueError("the sampling strategy should be either 'ddim' or 'ddpm'")

    def sample_prev_time_ddim(self,
        xt:torch.Tensor,
        noise_pred:torch.Tensor,
        t:int
    ):
        r"""Sample `x(t-1)` given `x(t)` and `noise_pred` the noise predicted by model.
        We follow the deterministic DDIM backward process.
        This allows `num_time_steps` to be smaller than the nb of steps in the forward process.
        
        Parameters
        ----------
        xt : torch.Tensor
            image at timestep `t`, of shape B x C x H x W
        noise_pred : torch.Tensor
            predicted noise, of shape B x C x H x W
        t : int
            current time step
        """
        mean = (xt - self.sqrt_one_minus_a_bar[t].to(xt.device) * noise_pred) / self.sqrt_a_bar[t].to(xt.device)
        if t == 0:
            return mean
        return self.sqrt_a_bar[t - 1].to(xt.device) * mean \
            + self.sqrt_one_minus_a_bar[t - 1].to(xt.device) * noise_pred

    def sample_prev_time_ddpm(self, xt, noise_pred, t):
        r"""Sample `x(t-1)` given `x(t)` and `noise_pred` the noise predicted by model.
        We follow the reverse dynamics of the forward process.
        
        Parameters
        ----------
        xt : torch.Tensor
            image at timestep `t`, of shape B x C x H x W
        noise_pred : torch.Tensor
            predicted noise, of shape B x C x H x W
        t : int
            current time step
        """
        # mean of x_(t-1)
        mean = xt - (1 - self.a.to(xt.device)[t]) * noise_pred / torch.sqrt(1 - self.a_bar.to(xt.device)[t])
        mean = mean / torch.sqrt(self.a.to(xt.device)[t])
        
        # only return mean
        if t == 0:
            return mean
        variance =  (1 - self.a_bar.to(xt.device)[t-1]) / (1 - self.a_bar.to(xt.device)[t])
        variance = variance * self.b.to(xt.device)[t]
        sigma = variance ** 0.5
        z = torch.randn(xt.shape).to(xt.device)
        return mean + sigma * z
    
def get_device():
    """Determine the gpu device available (if any) on the current machine.
    """
    if system() == 'Darwin':
        best_device = 'mps'
        is_available = torch.mps.is_available()
    elif system() == 'Windows':
        best_device = 'cuda'
        is_available = torch.cuda.is_available()
    return torch.device(best_device if is_available else 'cpu')

def train(
    dataset: torch.utils.data.Dataset,
    cfg_train: Config_training,
    model: torch.nn.Module,
    dfp: DiffusionForwardProcess
):
    r"""Train `model` according to the specifications in `cfg_train`.
    `dfp`is used to add noise to the samples, which are drawn from `dataset`.

    Each time the model achieves a better accuracy,
    it is saved at the path indicated in `cfg_train`.

    Returns
    -------
    logs : Dict[str, List]
        data collected to have some information about how the training has gone
    """
    # Dataloader
    dataloader = DataLoader(dataset, cfg_train.batch_size, shuffle=True, drop_last=True)
    
    # Device
    device = get_device()
    print(f'device: {device}\n')
    
    # Initiate Model
    model = model.to(device)
    
    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
    criterion = MSELoss()
    
    # Best Loss: used to decide when to save the model weights
    best_loss = float('inf')
    
    # metrics to return at the end of training
    logs = {
        "logloss": []
    }

    # Train
    for epoch in tqdm(range(cfg_train.n_epochs)):
        epoch_loss = []

        # Set model to train mode
        model.train()
        
        # Loop over dataloader
        for imgs in dataloader:
            imgs = imgs.to(device)
            
            # Generate noise and timestamps
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, dfp.n_steps, (cfg_train.batch_size,)).to(device)
            
            # Add noise to the images using Forward Process
            noisy_imgs = dfp.add_noise(imgs, noise, t)
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Predict noise using U-net Model
            noise_pred = model(noisy_imgs, t)
            
            # Calculate Loss
            loss = criterion(noise_pred, noise)
            epoch_loss.append(torch.log10(loss).item())
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()
        
        # Mean Log Loss
        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        
        """
        # Display
        print('Epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            mean_epoch_loss,
        ))
        """

        logs["logloss"].extend(epoch_loss)
        
        # model save based on train-loss
        if mean_epoch_loss < best_loss:
            best_loss = mean_epoch_loss
            torch.save(model, cfg_train.model_path)
    print('Done training.....')
    return logs

def generate(
    img_size: Tuple[int, int],
    model: torch.nn.Module,
    drp: DiffusionReverseProcess,
    device: torch.device
):
    r"""Generate an image from an initial Gaussian noise by using `model` to predict
    the noise at each time step, and using `drp` to infer the denoised sample at
    previous time step.

    Returns
    -------
    x0 : torch.Tensor
        the generated image, or the inferred initial condition
        of the forward diffusion process
    """
    # set model to eval mode
    model = model.to(device)
    model.eval()
    
    # generate noise sample
    xt = torch.randn(1, *img_size).to(device)
    
    # denoise step by step
    with torch.no_grad():
        for t in reversed(range(drp.n_backward_steps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt = drp.sample_prev_time(xt, noise_pred, torch.as_tensor(t).to(device))

    # get the image into the correct format
    xt = xt.detach().cpu().view(*img_size)
    
    return (xt + 1) / 2

def whole_train():
    config_dataset = Config_mnist()

    config_model = {
        "sample_size": 32,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": (64, 64, 64),
        "down_block_types": (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        "up_block_types": (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        )
    }

    dataset = datasets.CustomMnist(config_dataset)
    
    print()
    print("number of training samples:", len(dataset))
    print()

    model = models.CustomUnet(**config_model)
    print("nb of model parameters:", models.count_params(model))
    print()

    cfg_dfp = Config_dfp()
    cfg_train = Config_training()

    dfp = DiffusionForwardProcess(cfg_dfp)

    logs = train(dataset, cfg_train, model, dfp)

    # visualize how the training has gone
    prefix = "src_code/diffusion_model/figures/"
    _, ax = plt.subplots()
    ax.plot(logs["logloss"], '.')
    ax.set_xlabel("nb of gradient updates")
    ax.set_ylabel("log10 of the loss")
    plt.savefig(prefix + "logloss.png")

def whole_sample():
    # load the trained model
    model_path = Config_training.model_path
    model = torch.load(model_path, weights_only=False)

    # load the image size
    img_size = Config_mnist.img_size

    # generate some samples with the trained model
    prefix = "src_code/diffusion_model/figures/"

    # get the device
    device = get_device()

    # initialize the Diffusion Reverse Process
    drp = DiffusionReverseProcess(Config_drp())
    
    print()
    print("sampling from the trained model...")

    for k in tqdm(range(4)):
        datasets.display_mat(generate(img_size, model, drp, device))
        plt.savefig(prefix + str(k) + ".png")

if __name__ == "__main__":
    # will train a UNet2DModel on a small subset (8 samples) of the mnist dataset
    # whole_train()

    # will load the trained model and generate 4 samples according to
    # the deterministic DDIM sampling strategy
    # these samples are stored in "figures/"
    whole_sample()

    # visualize the images of the training set (8 here)
    config_dataset = Config_mnist()

    dataset = datasets.CustomMnist(config_dataset)
    for img in dataset:
        datasets.display_mat(img)
        plt.show()
