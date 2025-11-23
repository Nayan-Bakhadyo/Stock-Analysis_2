#!/bin/bash
# Run SPC prediction with best config

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate Stock_Prediction

# Run prediction
python3 predict_single_config.py
