#!/bin/bash

# Qwen3-Omni Setup Script for Linux
# This script sets up the virtual environment and installs dependencies

set -e  # Exit on any error

echo "================================================================================"
echo "Qwen3-Omni Setup for Linux"
echo "================================================================================"
echo

# Check if Python 3.8+ is available
python3 --version >/dev/null 2>&1 || { echo "Error: Python 3 is not installed or not in PATH"; exit 1; }

# Get Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

# Check if version is 3.8+
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "Error: Python 3.8+ is required. You have $PYTHON_VERSION"
    exit 1
fi

echo "Creating virtual environment..."
if [ -d "../.venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf ../.venv
fi

python3 -m venv ../.venv
source ../.venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch..."
# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "+ NVIDIA GPU detected"
    echo
    echo "Select PyTorch version:"
    echo "  1. CUDA 12.1 (RTX 30xx/40xx, newer GPUs) - RECOMMENDED"
    echo "  2. CUDA 12.9 (latest GPUs, RTX 50xx series)"
    echo "  3. CUDA 11.8 (older GPUs)"
    echo "  4. CPU only (no GPU)"
    echo
    read -p "Enter choice (1-4, default=1): " cuda_choice
    cuda_choice=${cuda_choice:-1}

    case $cuda_choice in
        1)
            echo "Installing PyTorch with CUDA 12.1..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        2)
            echo "Installing PyTorch with CUDA 12.9..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
            ;;
        3)
            echo "Installing PyTorch with CUDA 11.8..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        *)
            echo "Installing PyTorch CPU version..."
            pip install torch torchvision torchaudio
            ;;
    esac
else
    echo "! No NVIDIA GPU detected, installing CPU version"
    pip install torch torchvision torchaudio
fi

echo
echo "Installing other dependencies..."
pip install transformers accelerate qwen-omni-utils werpy tqdm pyyaml soundfile librosa scipy pandas numpy bitsandbytes

echo
echo "Installing FlashAttention 2 (optional but recommended)..."
pip install flash-attn --no-build-isolation || echo "FlashAttention installation failed, continuing..."

echo "+ Dependencies installed"
echo
echo "================================================================================"
echo "Setup Complete!"
echo "================================================================================"
echo
echo "Virtual environment created at: ../.venv"
echo "To activate: source ../.venv/bin/activate"
echo
echo "Next steps:"
echo "  1. Run: ./update_transformers.sh"
echo "  2. Run: ./download_model.sh"
echo "  3. Run: ./run_pipeline.sh"
echo
