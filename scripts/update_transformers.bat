@echo off
REM Update transformers for Qwen3-Omni support

echo ========================================
echo Update Transformers for Qwen3-Omni
echo ========================================
echo.

if not exist "..\\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    echo Please run setup.bat first
    exit /b 1
)

call ..\\.venv\Scripts\activate.bat

echo Qwen3-Omni requires transformers from GitHub source.
echo.
echo This will:
echo   1. Uninstall current transformers
echo   2. Install transformers from GitHub
echo   3. Install qwen-omni-utils
echo   4. Install flash-attn (optional, may take time)
echo.
pause

echo.
echo [1/4] Uninstalling current transformers...
pip uninstall transformers -y

echo.
echo [2/4] Installing transformers from GitHub...
pip install git+https://github.com/huggingface/transformers

echo.
echo [3/4] Installing qwen-omni-utils...
pip install qwen-omni-utils -U

echo.
echo [4/4] Installing flash-attn (optional, may fail on some systems)...
pip install flash-attn --no-build-isolation
if %errorlevel% neq 0 (
    echo ! FlashAttention installation failed - this is optional
    echo Model will work without it, just slower
)

echo.
echo ========================================
echo Update Complete!
echo ========================================
echo.
echo Test the installation:
echo   cd ..
echo   python -c "from transformers import AutoModel; print('Success!')"
echo.
pause

