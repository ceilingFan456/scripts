#!/usr/bin/env bash
set -e

source secret.sh

ENV_NAME=showo
WORKDIR=$HOME/workspace
MAX_JOBS=16

# -------- system deps --------
sudo apt-get update
sudo apt-get install -y \
    git git-lfs curl ca-certificates \
    build-essential cmake ninja-build pkg-config \
    libcurl4-openssl-dev \
    libssl-dev libffi-dev \
    python3-dev

git lfs install

# -------- conda env --------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n $ENV_NAME python=3.10
conda activate $ENV_NAME

export MAX_JOBS=$MAX_JOBS

python -m pip install -U pip wheel setuptools

# -------- torch cu121 --------
pip install \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# -------- clone repos --------
mkdir -p $WORKDIR
cd $WORKDIR

git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/show-o-dev.git
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedEValKit-dev.git
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedSAM.git

# -------- install show-o --------
pip install -r show-o-dev/requirements.txt
pip install --no-build-isolation flash-attn==2.5.7
pip install -e show-o-dev

# -------- install MedEvalKit --------
pip install -r MedEValKit-dev/medevalkit/requirements.txt

pip install \
    "networkx>=3.4.2" \
    'open_clip_torch[training]' \
    datasets google-generativeai \
    openai azure-identity qwen-vl-utils \
    nltk rouge mathruler tenacity pylatexenc pydash

pip install -r MedEValKit-dev/medevalkit/requirements_additional.txt
pip install -e MedEValKit-dev/medevalkit/LLaVA-NeXT
pip install -e MedEValKit-dev

# -------- install MedSAM --------
pip install -e MedSAM

# -------- pin numpy --------
pip install numpy==1.24.4

# -------- vllm --------
pip install vllm

echo "DONE. Run:"
echo "source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME"
