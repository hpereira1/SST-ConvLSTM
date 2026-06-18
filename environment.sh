#!/bin/bash
# Setup script for SST prediction project

echo "=========================================="
echo "SST Prediction Project Setup"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo ""
echo "Creating conda environment 'sstml'..."
conda env create -f environment.yml

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate sstml"
echo ""
echo "To start training, run:"
echo "  python train.py --data-path data/raw/your_sst_data.nc"
echo ""
echo "For more information, see README.md"

