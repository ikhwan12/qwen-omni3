@echo off
REM Complete installation of all dependencies for Qwen3-Omni

echo ================================================================================
echo Complete Installation for Qwen3-Omni Transcription System
echo ================================================================================
echo.

if not exist "..\\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    echo Please run setup.bat first
    exit /b 1
)

call ..\\.venv\Scripts\activate.bat

echo This will install/update all dependencies for Qwen3-Omni:
echo.
echo   1. PyTorch with CUDA 12.1
echo   2. Transformers from GitHub (Qwen3-Omni support)
echo   3. qwen-omni-utils
echo   4. flash-attn
echo   5. werpy and other dependencies
echo.
echo This may take 10-15 minutes...
echo.
pause

echo.
echo ================================================================================
echo [1/5] Installing PyTorch with CUDA 12.1
echo ================================================================================
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ================================================================================
echo [2/5] Installing Transformers from GitHub
echo ================================================================================
pip uninstall transformers -y
pip install git+https://github.com/huggingface/transformers

echo.
echo ================================================================================
echo [3/5] Installing qwen-omni-utils
echo ================================================================================
pip install qwen-omni-utils -U

echo.
echo ================================================================================
echo [4/5] Installing flash-attn (may take several minutes)
echo ================================================================================
pip install flash-attn --no-build-isolation
if %errorlevel% neq 0 (
    echo.
    echo ! FlashAttention installation failed - this is OPTIONAL
    echo ! Model will work without it, just slightly slower
    echo ! You can skip this if it keeps failing
    echo.
)

echo.
echo ================================================================================
echo [5/5] Installing other dependencies
echo ================================================================================
pip install werpy accelerate tqdm pyyaml soundfile librosa pandas numpy

echo.
echo ================================================================================
echo Installation Complete!
echo ================================================================================
echo.
echo Now verify:
echo   cd ..
echo   test_cuda.bat
echo.
echo Then test transcription:
echo   python transcribe.py --speaker ABA
echo.
pause

