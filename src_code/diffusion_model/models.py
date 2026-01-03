"""Gathers various models used to denoise Gaussian white noise back to some target distribution.

For the implementation of the U-net architecture with time embedding, we use the implementation of HuggingFace.
"""
from diffusers import UNet2DModel
import torch

def count_params(
    model: torch.nn.Module
):
    """Return the number of parameters in `model`.

    Parameters
    ----------
    model : torch.nn.Module
        the model we want the number of parameters of
    """
    nb_params = 0
    for param in model.parameters():
        nb_params += torch.prod(torch.tensor(param.shape))
    return nb_params.item()

class CustomUnet(UNet2DModel):
    def __init__(self, **kwargs):
        UNet2DModel.__init__(self, **kwargs)
    def forward(self, sample: torch.Tensor, timestep):
        """Sample is supposed to be of shape (batch_size, *img_size).
        """
        shape = sample.shape
        return super().forward(
            sample.view(shape[0], 1, shape[-2], shape[-1]),
            timestep, return_dict = False)[0].view(shape[0], shape[-2], shape[-1])

if __name__ == "__main__":
    config_model = {
        "sample_size": (32, 32),
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 2, # how many ResNet layers to use per UNet block
        "block_out_channels": (64, 64, 64),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
        ),
        "mid_block_type": (
            None,
        ),
        "up_block_types": (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        "norm_num_groups": 8
    }

    model = UNet2DModel(**config_model)
    
    print("nb of parameters:", count_params(model))
    exit()

    batch_size = 8
    nb_channels = 1
    imgs = torch.rand((batch_size, nb_channels, *config_model["sample_size"]))
    timesteps = torch.randint(0, 17, (batch_size,))

    test = model(imgs, timesteps, return_dict=False)
    print(test[0].shape)
