#!/bin/bash

# Check CUDA availability and GPU details
set -e

echo "================================================================================"
echo "CUDA and GPU Check"
echo "================================================================================"
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Go to parent directory
cd ..

echo "Checking CUDA availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
else:
    print('CUDA not available - will use CPU')
"

echo
echo "Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found - NVIDIA drivers may not be installed"
fi

echo
echo "Checking if model will use CUDA..."
python scripts/test_model_cuda.py
