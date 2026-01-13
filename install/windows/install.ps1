# generate the folder structure "figures" "models_saved" "train_data"
$prefix="..\..\src_code\diffusion_model\"

$names = 'test1','test2'
foreach ($name in $names) {
    if (!(& test-path -PathType "container" "${prefix}$name"))
    {
        & New-Item -ItemType "Directory" -Path "${prefix}$name"
    }
}

# create a virtual environment for Python:
# the folder .venv must be at the project root
prefix="..\..\"
& py -m venv "${prefix}.venv"
& $prefix.venv\Scripts\activate

# prepare pip
& py -m pip install --upgrade pip

# install the Python librairies
& py -m pip install -r "../requirements.txt"
