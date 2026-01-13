"""
Implement a simple diffusion model using backward Langevin dynamics and a score vector function learned by a neural network (differential model).
"""
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from platform import system

from diffusion_model.configurations import Config_ddpm, Config_mnist, Config_training
import diffusion_model.models as models
import diffusion_model.datasets as datasets

class DiffusionForwardProcess:
    r"""
    Forward Process class as described in the 
    paper "Denoising Diffusion Probabilistic Models"
    """
    
    def __init__(self,
        cfg: Config_ddpm
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
        r"""Adds noise to a batch of original images at time-step t.
        
        :param original: Input Image Tensor
        :param noise: Random Noise Tensor sampled from Normal Dist N(0, 1)
        :param t: timestep of the forward process of shape -> (B, )
        
        Notes
        -----
        time-step t may differ for each image inside the batch.
        """
        return self.sqrt_a_bar.to(original.device)[t][:, None, None] * original \
            + self.sqrt_one_minus_a_bar.to(original.device)[t][:, None, None] * noise

class DiffusionReverseProcess:
    r"""Reverse Process class as described in the 
    paper "Denoising Diffusion Probabilistic Models"
    """
    
    def __init__(self, 
        num_time_steps = 1000, 
        beta_start = 1e-4, 
        beta_end = 0.02
    ):
        # Precomputing beta, alpha, and alpha_bar for all t's.
        self.b = torch.linspace(beta_start, beta_end, num_time_steps)
        self.a = 1 - self.b
        self.a_bar = torch.cumprod(self.a, dim=0)
        self.sqrt_a_bar = torch.sqrt(self.a_bar)
        self.sqrt_one_minus_a_bar = torch.sqrt(1 - self.a_bar)

    def sample_prev_time_ddim(self, xt, noise_pred, t):
        r"""Sample x_(t-1) given x_t and noise predicted by model, by following the dynamics
        of the deterministi DDIM backward process.
        This allows `num_time_steps` to be smaller than the nb of steps in the forward process.
        
        :param xt: Image tensor at timestep t of shape -> B x C x H x W
        :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
        :param t: Current time step
        """
        mean = (xt - self.sqrt_one_minus_a_bar[t].to(xt.device) * noise_pred) / self.sqrt_a_bar[t].to(xt.device)
        if t == 0:
            return mean
        return self.sqrt_a_bar[t - 1].to(xt.device) * mean \
            + self.sqrt_one_minus_a_bar[t - 1].to(xt.device) * noise_pred

    def sample_prev_time_ddpm(self, xt, noise_pred, t):
        r"""Sample x_(t-1) given x_t and noise predicted by model, by following the reverse dynamics
        of the forward process.
        
        :param xt: Image tensor at timestep t of shape -> B x C x H x W
        :param noise_pred: Noise Predicted by model of shape -> B x C x H x W
        :param t: Current time step
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

def generate(cfg: Config_ddpm, model, drp, device):
    """Given Pretrained DDPM U-net model, Generate Real-life
    Images from noise by going backward step by step. i.e.,
    Mapping of Random Noise to mnist images.
    """
    # Set model to eval mode
    model = model.to(device)
    model.eval()
    
    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.img_size, cfg.img_size).to(device)
    
    # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.n_backward_steps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt = drp.sample_prev_time_ddim(xt, noise_pred, torch.as_tensor(t).to(device))

    # Convert the image to proper scale
    # xt = torch.clamp(xt, -1., 1.).detach().cpu().view(cfg.img_size, cfg.img_size)
    xt = xt.detach().cpu().view(cfg.img_size, cfg.img_size)
    
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

    cfg_ddpm = Config_ddpm()
    cfg_train = Config_training()

    dfp = DiffusionForwardProcess(cfg_ddpm)

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
    model_path = Config_training().model_path
    model = torch.load(model_path, weights_only=False)

    # generate some samples with the trained model
    prefix = "src_code/diffusion_model/figures/"

    # Device
    device = get_device()
    cfg = Config_ddpm()

    # Initialize Diffusion Reverse Process
    drp = DiffusionReverseProcess(cfg.n_backward_steps)
    
    print()
    print("sampling from the trained model...")

    for k in tqdm(range(4)):
        datasets.display_mat(generate(cfg, model, drp, device))
        plt.savefig(prefix + str(k) + ".png")

if __name__ == "__main__":
    # whole_train()

    whole_sample()

    # print the images of the training set
    """
    config_dataset = Config_mnist()

    dataset = datasets.CustomMnist(config_dataset)
    for img in dataset:
        datasets.display_mat(img)
        plt.show()
    """
