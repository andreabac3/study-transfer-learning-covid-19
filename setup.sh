#!/bin/bash

# setup conda
source ~/miniconda3/etc/profile.d/conda.sh

# create conda env
read -rp "Enter environment name: " env_name
echo "In our experiments we used the python version: 3.7.11"
read -rp "Enter python version (e.g. 3.7.11) " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# install torch
read -rp "Enter cuda version (e.g. 10.1 or none to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch==1.9 torchvision==0.10.0 cpuonly -c pytorch
else
    conda install -y pytorch==1.9 torchvision==0.10.0 cudatoolkit=$cuda_version -c pytorch
fi

# install python requirements
pip install -r requirements.txt

touch .env
echo PROJECT_ROOT="$(pwd)" >> .env