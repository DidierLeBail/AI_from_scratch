# AI from scratch
This repo contains some basic implementations of some of the commonly used concepts in modern deep learning.
(I plan to enrich it with other implementations in the future).
It is organized in three directories:

## install/
Contains executables to make the source code work.
To install the code, follow these steps:

1. install Python 3.8+ (and git)
2. clone this repo on your computer
3. open a terminal and go the folder "install/macos" or "install/windows" depending on your OS
4. run "install.sh" or "install.ps1": it will create the appropriate folder structure and a virtual environment for Python, that will gather all the packages you need to run the source code

## docs/
Allows to generate a documentation html from the formatted strings present in the code.
To use it, open a terminal and go to the directory "docs/" and execute "make_doc.sh".
To do that e.g. in MacOS, you can run
```shell
zsh make_doc.sh
```
Once the command has completed, you will the doc by opening the file "docs/build/html/index.html".
Run the same command each time you want to rebuild the doc.

## src_code/
Contains the source code.
Two directories can be found there:

[diffusion_model/](https://github.com/DidierLeBail/AI_from_scratch/tree/main/src_code/diffusion_model)

Here an implementation of a diffusion model, where a [UNet2DModel]() is trained as a denoiser.
The current implementation is tested on the MNIST dataset.

[transformer/](https://github.com/DidierLeBail/AI_from_scratch/tree/main/src_code/transformer)

Here an implementation of the multi-head attention mechanism, of the encoder part of the original Transformer architecture, and a custom tokenizer, using the byte-pair encoding algorithm (except it acts on strings and not byte streams).
