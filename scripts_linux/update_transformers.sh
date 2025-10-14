#!/bin/bash

# Update transformers to support Qwen3-Omni
set -e

echo "================================================================================"
echo "Updating Transformers for Qwen3-Omni Support"
echo "================================================================================"
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

echo "Uninstalling existing transformers..."
pip uninstall -y transformers

echo "Installing transformers from GitHub (with Qwen3-Omni support)..."
pip install git+https://github.com/huggingface/transformers

echo "Installing qwen-omni-utils..."
pip install qwen-omni-utils

echo "Installing FlashAttention 2..."
pip install flash-attn --no-build-isolation || echo "FlashAttention installation failed, continuing..."

echo
echo "================================================================================"
echo "Transformers Updated!"
echo "================================================================================"
echo
echo "Qwen3-Omni support is now available."
echo "You can now download and use Qwen3-Omni models."
echo
