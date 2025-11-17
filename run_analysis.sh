#!/bin/bash
# Run stock analysis with proper conda environment

source ~/anaconda3/etc/profile.d/conda.sh
conda activate Stock_Prediction
python3 simple_analysis.py NABIL NICA SCB
