#!/usr/bin/env bash
set -e

source secret.sh

ENV_NAME=showo
WORKDIR=$HOME/workspace
MAX_JOBS=16
CONDA_DIR=$HOME/miniconda

# -------- system deps --------
sudo apt-get update
sudo apt-get install -y \
    git git-lfs curl ca-certificates wget \
    build-essential cmake ninja-build pkg-config \
    libcurl4-openssl-dev \
    libssl-dev libffi-dev \
    python3-dev

git lfs install

# -------- install conda (Miniconda) --------
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_DIR
    rm miniconda.sh
    # Add to path for this current session
    export PATH="$CONDA_DIR/bin:$PATH"
fi

# -------- conda env --------
# Ensure conda commands work in this subshell
source "$CONDA_DIR/etc/profile.d/conda.sh" || source "$(conda info --base)/etc/profile.d/conda.sh"

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create env if it doesn't exist
if ! conda info --envs | grep -q "$ENV_NAME"; then
    conda create -y -n $ENV_NAME python=3.10
fi

conda activate $ENV_NAME

export MAX_JOBS=$MAX_JOBS

# Update pip first
python -m pip install -U pip wheel setuptools

# -------- torch cu121 --------
pip install \
    torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121

# -------- clone repos --------
mkdir -p $WORKDIR
cd $WORKDIR

# Note: Ensure GITHUB_TOKEN is exported in secret.sh
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/show-o-dev.git
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedEValKit-dev.git
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedSAM.git

# -------- install show-o --------
pip install -r show-o-dev/requirements.txt
# Flash-attn can take a long time to compile
# pip install --no-build-isolation flash-attn==2.5.7
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
# Note: Pinning after other installs to ensure it overrides dependencies
pip install numpy==1.24.4

# -------- vllm --------
pip install vllm

## reinstall to correct version issues. 
pip install -r show-o-dev/requirements.txt
# Flash-attn can take a long time to compile
pip install --no-build-isolation flash-attn==2.5.7

echo "DONE. Run:"
echo "source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME"

pwd

cd /scratch/amlt_code

touch test_file.txt

echo "Setup complete. Job is now sleeping indefinitely to allow for SSH access..." >> test_file.txt
