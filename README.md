# code-for-AortaSeg24

# Installation
Requirements: Ubuntu 20.04, CUDA 11.8

Create a virtual environment: conda create -n Aorta_mamba python=3.10 -y and conda activate Aorta_mamba 
Install Pytorch 2.0.1: pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
Install Mamba: pip install causal-conv1d>=1.2.0 and pip install mamba-ssm --no-cache-dir
Download code: https://github.com/yuejinghua/code-for-AortaSeg24.git​. 
cd mamba_gag and run pip install -e .

# How to work?
Data storage and pre-processing according to nnunetv2 framework 
https://github.com/MIC-DKFZ/nnUNet.git

Training
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD  -tr nnUNetTrainerUME_LGAG

Inference
python nnunetv2/inference/predict_from_raw_data_AortaSeg2024_cf.py

