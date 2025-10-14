@echo off
REM Quick CUDA test script

echo ========================================
echo Quick PyTorch CUDA Test
echo ========================================
echo.

if not exist "..\\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    echo Please run setup.bat first
    exit /b 1
)

call ..\\.venv\Scripts\activate.bat

echo [Test 1] Checking if CUDA is available...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo.

echo [Test 2] Checking CUDA version...
python -c "import torch; print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
echo.

echo [Test 3] Checking GPU name...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
echo.

echo [Test 4] Testing CUDA tensor operation...
python -c "import torch; x = torch.rand(3, 3).cuda() if torch.cuda.is_available() else torch.rand(3, 3); print('Test tensor created on:', x.device); print('Tensor:\n', x)"
echo.

echo [Test 5] Checking PyTorch build...
python -c "import torch; print('PyTorch version:', torch.__version__)"
echo.

echo ========================================
echo Test Complete
echo ========================================
pause

