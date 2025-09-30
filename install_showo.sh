git clone https://github.com/ceilingFan456/Show-o.git

cd Show-o

conda create -n showo python=3.10 -y

conda activate showo

sudo apt update
sudo apt install libcurl4-openssl-dev
pip install -r requirements.txt