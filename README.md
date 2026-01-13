# AI from scratch
This repo contains simple implementations of some of the commonly used concepts in modern deep learning.
(I plan to enrich it with other implementations in the future).
It is organized in three directories:

## install/
Contains executables to make the source code usable on your machine.
To install the code, follow these steps:

1. install Python 3.8+ (and git)
2. clone this repo on your computer
3. open a terminal and go to the directory "install/macos" or "install/windows" depending on your OS (the installation for windows has not been tested yet)
4. run "install.sh" or "install.ps1": it will create the appropriate folder structure and a virtual environment for Python, that will gather all the packages you need to run the source code

## src_code/
Contains the source code.
Two directories can be found there:

### diffusion_model/

Here an implementation of a diffusion model, where a [UNet2DModel](https://huggingface.co/docs/diffusers/api/models/unet2d) is trained as a denoiser.
The current implementation is tested on the MNIST dataset.

### transformer/

Here an implementation of the multi-head attention mechanism, of the encoder part of the original Transformer architecture, and a tokenizer, implementing the byte-pair encoding algorithm (except it parses strings instead of byte streams).

## docs/
Do not try to build the doc by yourself.
Just open the file "docs/build/html/index.html" in a web browser.
