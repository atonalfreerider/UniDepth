#!/bin/bash
conda create -y -n Unidepth python=3.11
conda activate Unidepth

pip install torch==2.4.1 torchvision torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
