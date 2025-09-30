#!/usr/bin/env bash
set -e

# Install location (persistent storage)
INSTALL_DIR="$HOME/disk/miniconda3"

# Clean up any previous install
rm -rf "$INSTALL_DIR"

# Download latest Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# Install Miniconda silently
bash /tmp/miniconda.sh -b -p "$INSTALL_DIR"

# Add conda to PATH in ~/.bashrc
if ! grep -q "$INSTALL_DIR/bin" ~/.bashrc; then
  echo 'export PATH="'"$INSTALL_DIR"'/bin:$PATH"' >> ~/.bashrc
fi

# Apply changes to current shell
export PATH="$INSTALL_DIR/bin:$PATH"

# Verify install
conda --version

# 1) Load Conda’s shell hook now (enables `conda create/activate` in THIS shell)
eval "$($HOME/disk/miniconda3/bin/conda shell.bash hook)"

# 2) Make it permanent for future shells
$HOME/disk/miniconda3/bin/conda init bash

source ~/.bashrc

