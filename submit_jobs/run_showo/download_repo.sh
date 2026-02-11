#!/bin/bash

source /mnt/almt_code/secret.sh

echo "Cloning Show-o repository..."
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/show-o-dev.git

echo "Cloning MedEValKit repository..."
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedEValKit-dev.git

echo "Cloning MedSAM repository..."
git clone https://$GITHUB_TOKEN@github.com/ceilingFan456/MedSAM.git