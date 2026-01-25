#!/bin/bash

## download the repo
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive

## create conda env 
conda create -n openpi_env python=3.11 -y
conda activate openpi_env

## download uv 
## https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

## install dependencies
## there is a uv.lock file for you to refer to. 
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
uv pip install pytest


## install google cloud sdk 
## https://docs.cloud.google.com/sdk/docs/install-sdk#linux
cd ~
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz
tar -xvf google-cloud-cli-linux-x86_64.tar.gz
echo 'export PATH="$HOME/google-cloud-sdk/bin:$PATH"' >> ~/.bashrc


## download checkpoints
cd ~/code/openpi/checkpoints
gcloud alpha storage cp -r gs://openpi-assets/checkpoints/pi05_base .