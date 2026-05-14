#!/bin/bash

# MDR Project - Random Forest Environment Setup Script
# This script sets up the Python environment for the Random Forest model

set -e  # Exit on any error

echo "=========================================="
echo "MDR Project - Environment Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Project paths
PROJECT_ROOT="/Users/kerrycheon/PycharmProjects/MDR-Project"
VENV_PATH="$PROJECT_ROOT/Temporal/Pipeline/venv"
PYTHON_BIN="$VENV_PATH/bin/python"
PIP_BIN="$VENV_PATH/bin/pip"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}Virtual environment: $VENV_PATH${NC}"
echo ""

# Check if venv exists
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PATH${NC}"
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment found${NC}"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
$PYTHON_BIN -m pip install --upgrade pip setuptools wheel

# Install essential packages for Random Forest notebook
echo ""
echo "Installing essential packages..."
$PIP_BIN install \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    notebook \
    ipykernel \
    ipywidgets \
    tqdm \
    openpyxl

echo -e "${GREEN}✓ Essential packages installed${NC}"

# Install ipykernel and register kernel
echo ""
echo "Registering Jupyter kernel..."
$PYTHON_BIN -m ipykernel install --user --name=mdr-pipeline --display-name="MDR Pipeline (Python 3.14)"
echo -e "${GREEN}✓ Jupyter kernel registered${NC}"

# Verify installation
echo ""
echo "Verifying installation..."
$PYTHON_BIN -c "
import sklearn
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
import jupyter
import tqdm

print('Package versions:')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  pandas: {pd.__version__}')
print(f'  numpy: {np.__version__}')
print(f'  matplotlib: {matplotlib.__version__}')
print(f'  seaborn: {sns.__version__}')
"

echo ""
echo -e "${GREEN}=========================================="
echo -e "✓ Setup Complete!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo ""
echo "Available kernels:"
jupyter kernelspec list
echo ""
echo -e "${BLUE}To activate the virtual environment, run:${NC}"
echo "  source $VENV_PATH/bin/activate"
echo ""
echo -e "${BLUE}To run the notebook in VSCode:${NC}"
echo "  1. Open: Models/Temporal/v6/v6.1/MDR-v6.1-RandomForest.ipynb"
echo "  2. Select kernel: MDR Pipeline (Python 3.14)"
echo ""
echo -e "${BLUE}To run the notebook in Jupyter:${NC}"
echo "  source $VENV_PATH/bin/activate"
echo "  jupyter notebook Models/Temporal/v6/v6.1/MDR-v6.1-RandomForest.ipynb"
echo ""
echo -e "${GREEN}Happy modeling! 🚀${NC}"
