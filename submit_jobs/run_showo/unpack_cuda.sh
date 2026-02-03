#!/bin/bash

source secret.sh

## extracting the saved conda pack
mkdir -p /tmp/eval
tar -xzf /mnt/default_storage/qiming/envs/eval_pack_ND96_H100_v5_cuda12.8_py310_v1.tar.gz \
    -C /tmp/eval
source /tmp/eval/bin/activate

## install all repos 
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/show-o-dev.git
cd show-o-dev/
pip install -e .
cd ..

git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedEValKit-dev.git
cd MedEValKit-dev/
pip install -e .
cd medevalkit
cd LLaVA-NeXT && pip install -e .
cd ../..

git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedSAM.git
cd MedSAM/
pip install -e .
cd ..

## uninstall flash attention to avoid conflicts
pip uninstall -y flash-attn
pip install flash-attn==2.5.7 --no-build-isolation

## done with loading the conda. 