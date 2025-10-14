@echo off
REM Fix PyTorch CUDA installation

echo ========================================
echo Fix PyTorch CUDA Installation
echo ========================================
echo.

if not exist "..\\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    exit /b 1
)

call ..\\.venv\Scripts\activate.bat

echo Your GPU: Quadro RTX 8000 (48GB VRAM)
echo Current PyTorch: CPU-only version
echo.
echo This will:
echo   1. Uninstall CPU version of PyTorch
echo   2. Install PyTorch with CUDA 12.1 support
echo   3. Verify CUDA is working
echo.
pause

echo.
echo [Step 1/3] Uninstalling CPU version...
pip uninstall torch torchvision torchaudio -y

echo.
echo [Step 2/3] Installing CUDA version (this may take a few minutes)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [Step 3/3] Verifying installation...
python -c "import torch; print('='*60); print('CUDA Available:', torch.cuda.is_available()); print('PyTorch Version:', torch.__version__); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'); print('GPU Memory:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB' if torch.cuda.is_available() else 'N/A'); print('='*60)"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Run this to verify:
echo   test_cuda_quick.bat
echo.
pause

