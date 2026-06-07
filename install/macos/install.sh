#!/bin/sh

# generate the folder structure
prefix="../../src_code/diffusion_model/"
for name in "figures" "models_saved" "train_data"
do
    if [ ! -d $prefix$name ]; then
        mkdir $prefix$name
    fi
done

# create a virtual environment for Python:
# the folder .venv must be at the project root
prefix="../../"
python3 -m venv $prefix.venv
source $prefix.venv/bin/activate

# prepare pip
python3 -m pip install --upgrade pip

# install the Python librairies
python3 -m pip install -r "../requirements.txt"
