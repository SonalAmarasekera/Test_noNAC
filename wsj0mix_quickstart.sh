#!/bin/bash
set -e

# WSJ0 root data download and move
pip install kagglehub
python wsj0mix_kagglehub.py

# Cloning the needed repos
git clone https://github.com/mpariente/pywsj0-mix.git
git clone https://github.com/BlinkDL/RWKV-LM.git

# Required packages
pip install ninja wandb pandas scipy numpy
pip install git+https://github.com/descriptinc/descript-audio-codec

#Moving to RWKV directory and installing dependencies
cd RWKV-LM/RWKV-v7/train_temp
rm -f src/model.py
pip install -r requirements.txt
cd ../../../

# Renaming RWKV repo for access inside script
mv RWKV-LM/RWKV-v7 RWKV-LM/RWKV_v7
mv RWKV-LM RWKV

# Moving the edited model.py inside the original repo and cuda folder to the main area
mv model.py RWKV/RWKV_v7/train_temp/src/
cp -r RWKV/RWKV_v7/train_temp/cuda .

# Creating the WSJ0mix
cd pywsj0-mix/

echo "Starting data creation..."
# Generate data for 2 speakers at 8k
python generate_wsjmix.py -p /root/.cache/kagglehub/datasets/sonishmaharjan555/wsj0-2mix/versions/2/ -o /workspace/ -n 2 -sr 16000 --len_mode min

cd ../

# Mapping the CSVs
python make_csv.py --root /workspace/2speakers/wav16k/min/tr --out train_min.csv
python make_csv.py --root /workspace/2speakers/wav16k/min/cv --out dev_min.csv
python make_csv.py --root /workspace/2speakers/wav16k/min/tt --out test_min.csv

#python make_csv.py --root /workspace/2speakers/wav16k/max/tr --out train_max.csv
#python make_csv.py --root /workspace/2speakers/wav16k/max/cv --out dev_max.csv
#python make_csv.py --root /workspace/2speakers/wav16k/max/tt --out test_max.csv

# Connecting the latent generating files
chmod +x gen_latents.sh

./gen_latents.sh
