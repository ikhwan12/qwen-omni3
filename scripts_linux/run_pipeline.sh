#!/bin/bash

# Run the complete transcription and evaluation pipeline
set -e

echo "================================================================================"
echo "Qwen3-Omni Transcription Pipeline"
echo "================================================================================"
echo

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Go to parent directory
cd ..

echo "Step 1: Preparing dataset..."
python scripts/prepare_dataset.py

echo
echo "Step 2: Running transcription..."
echo "Starting with speaker ABA (you can change this)..."
python transcribe.py --speaker ABA

echo
echo "Step 3: Evaluating WER..."
python evaluate_wer.py --speaker ABA

echo
echo "================================================================================"
echo "Pipeline Complete!"
echo "================================================================================"
echo
echo "Results saved in outputs/ directory"
echo "Check outputs/transcription_results_ABA.json for transcription results"
echo "Check outputs/evaluation_results_ABA.json for WER evaluation"
echo
