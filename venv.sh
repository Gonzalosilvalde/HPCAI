#!/bin/bash

module load cesga/2025

rm -rf venv

python -m venv venv

source ./venv/bin/activate

# Los he sacado del ejemplo del CFR24
echo "Installing modules..."

python3 -m pip install tensorflow[and-cuda]
pip install jupyterlab-nvdashboard
pip install -U tensorboard-plugin-profile
pip install nvidia-pyindex
pip install nvidia-cuda-cupti
pip install matplotlib
pip install portpicker
pip install tensorflow-datasets

# Estos me dice chatgpt que los use tb
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm tensorboard matplotlib portpicker
