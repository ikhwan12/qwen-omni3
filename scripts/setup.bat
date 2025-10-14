@echo off
REM Setup script for Qwen3-Omni Transcription project
REM Windows Batch Script

echo ========================================
echo Qwen3-Omni Setup Script
echo ========================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python not found. Please install Python 3.8 or higher
    exit /b 1
)
python --version
echo + Python found
echo.

REM Create virtual environment (in parent directory)
echo [2/5] Creating virtual environment...
if exist "..\\.venv" (
    echo Virtual environment already exists
    set /p response="Do you want to recreate it? (y/N): "
    if /i "%response%"=="y" (
        rmdir /s /q ..\\.venv
        python -m venv ..\\.venv
        echo + Virtual environment recreated
    )
) else (
    python -m venv ..\\.venv
    echo + Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call ..\\.venv\Scripts\activate.bat
echo + Virtual environment activated
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
echo + pip upgraded
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
echo This may take several minutes...
echo.
echo Checking for CUDA support...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo + NVIDIA GPU detected
    echo.
    echo Select PyTorch version:
    echo   1. CUDA 12.1 ^(RTX 30xx/40xx, newer GPUs^) - RECOMMENDED
    echo   2. CUDA 11.8 ^(older GPUs^)
    echo   3. CPU only ^(no GPU^)
    echo.
    set /p cuda_choice="Enter choice (1-3, default=1): "
    if "%cuda_choice%"=="" set cuda_choice=1
    
    if "%cuda_choice%"=="1" (
        echo Installing PyTorch with CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "%cuda_choice%"=="2" (
        echo Installing PyTorch with CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo Installing PyTorch CPU version...
        pip install torch torchvision torchaudio
    )
) else (
    echo ! No NVIDIA GPU detected, installing CPU version
    pip install torch torchvision torchaudio
)
echo.

echo Installing other dependencies...
pip install transformers accelerate qwen-omni-utils werpy tqdm pyyaml soundfile librosa pandas numpy
echo.
echo Installing FlashAttention 2 (optional but recommended)...
pip install flash-attn --no-build-isolation
if %errorlevel% neq 0 (
    echo X Failed to install dependencies
    exit /b 1
)
echo + Dependencies installed
echo.

echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo   1. Place your L2-ARCTIC dataset in the 'l2arctic' directory
echo   2. Edit 'config.yaml' to configure model and paths
echo   3. Run: .venv\Scripts\activate.bat
echo   4. Run: python transcribe.py
echo   5. Run: python evaluate_wer.py
echo.
echo Or use the automated pipeline:
echo   run_pipeline.bat
echo.
pause

