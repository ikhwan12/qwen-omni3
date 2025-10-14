#!/bin/bash

# Download Qwen3-Omni model
set -e

echo "================================================================================"
echo "Qwen3-Omni Model Download"
echo "================================================================================"
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Set cache directory to avoid C: drive issues (if on Windows/WSL)
export HF_HOME="$HOME/huggingface_cache"
mkdir -p "$HF_HOME"

echo "Cache location: $HF_HOME"
echo "Downloading to avoid space issues on system drive..."
echo

echo "Installing hf_transfer for faster downloads..."
pip install hf_transfer

echo "Downloading Qwen3-Omni-30B-A3B-Instruct model..."
echo "This is a large model (~75GB) and may take several hours."
echo

# Use huggingface-cli with resume support
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct --resume-download

if [ $? -eq 0 ]; then
    echo
    echo "================================================================================"
    echo "Download Complete!"
    echo "================================================================================"
    echo
    echo "Model is ready at: $HF_HOME/hub"
    echo
    echo "Now you can run:"
    echo "  cd .."
    echo "  python transcribe.py --speaker ABA"
    echo
else
    echo
    echo "================================================================================"
    echo "Download Failed!"
    echo "================================================================================"
    echo
    echo "Try running this script again - it will resume from where it left off."
    echo "If problems persist, check your internet connection and disk space."
    echo
    exit 1
fi
