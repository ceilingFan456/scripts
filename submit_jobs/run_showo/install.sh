#!/bin/bash

## install show-o first
export GITHUB_TOKEN=
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/show-o-dev.git

cd show-o-dev/
conda create -n showo python=3.10 -y
conda init 
source ~/.bashrc
conda activate showo

## install requirements
sudo apt update
sudo apt install -y libcurl4-openssl-dev
pip install -r requirements.txt
pip install --no-build-isolation --no-cache-dir "flash-attn==2.5.7"
pip install -e . 

cd ../
## done with installing showo 

## now install medevalkit
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedEValKit-dev.git

cd MedEValKit-dev/
conda create --name eval --clone showo
conda activate eval 

pip install -r medevalkit/requirements.txt
pip install "networkx>=3.4.2"
pip install 'open_clip_torch[training]'
python -m pip install datasets
python -m pip install google-generativeai

## should have flash-attn already from show-o env
# pip install flash-attn --no-cache-dir --no-build-isolation

# For LLaVA-like models
cd medevalkit
cd LLaVA-NeXT && pip install -e .
cd ../..

pip install -r medevalkit/requirements_additional.txt
pip install vllm
pip install openai
pip install azure-identity
python -m pip install qwen-vl-utils

python -m pip install \
  nltk \
  rouge \
  mathruler \
  openai \
  tenacity \
  azure-identity

python -m pip install pylatexenc
python -m pip install pydash

pip install -e .
cd .. 
## done with installing medevalkit


## install biomed sam 
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedSAM.git
cd MedSAM/

pip install -e .

## fix some problems at this point 
pip install numpy==1.24.4

cd .. 
## done with installing biomedsam


# ## optional to save the environment 
# ## naming convention <project>_<sku>_<cuda>_<python>_vN.tar.gz
# conda create -n eval_pack --clone eval
# conda activate eval_pack
# ## cannot pack with interactive packages 
# pip uninstall -y llava medevalkit medsam show_o
# conda pack \
#   -n eval_pack \
#   -o /mnt/default_storage/qiming/envs/eval_pack_ND96_H100_v5_cuda12.8_py310_v1.tar.gz

# ## can use this one for quick test
# mkdir -p /tmp/eval_pack_test
# tar -xzf /mnt/default_storage/qiming/envs/eval_pack_ND96_H100_v5_cuda12.8_py310_v1.tar.gz -C /tmp/eval_pack_test

# /tmp/eval_pack_test/bin/conda-unpack

# /tmp/eval_pack_test/bin/python - <<'PY'
# import numpy, torch
# print("numpy:", numpy.__version__)
# print("torch:", torch.__version__)
# print("cuda:", torch.cuda.is_available())
# print("gpus:", torch.cuda.device_count())
# PY
