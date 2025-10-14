@echo off
REM Simple model download without hf_transfer (more reliable)

echo ================================================================================
echo Download Qwen3-Omni-30B-A3B-Instruct Model (Simple Mode)
echo ================================================================================
echo.

if not exist "..\\.venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found
    exit /b 1
)

call ..\\.venv\Scripts\activate.bat

echo Model: Qwen/Qwen3-Omni-30B-A3B-Instruct
echo Size: ~75GB (15 files)
echo.
echo This uses standard download (no hf_transfer) - more reliable for unstable connections.
echo Downloads will resume automatically if interrupted.
echo.
echo Estimated time:
echo   - Fast internet: 2-4 hours
echo   - Medium internet: 4-8 hours
echo   - Slow internet: 8-16 hours
echo.
echo You can safely close this and run again later - it will resume!
echo.
pause

echo.
echo ================================================================================
echo Starting download...
echo ================================================================================
echo.

REM Set cache directory to D: drive to avoid C: drive space issues
set HF_HOME=D:\huggingface_cache
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

REM Disable hf_transfer to use standard download
set HF_HUB_ENABLE_HF_TRANSFER=0

echo Cache location: %HF_HOME%
echo Downloading to D: drive to avoid C: drive space issues...
echo.

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct', max_workers=4)"

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo Download Complete!
    echo ================================================================================
    echo.
    echo Model is ready at: %HF_HOME%\hub
    echo.
    echo Now you can run:
    echo   cd ..
    echo   python transcribe.py --speaker ABA
    echo.
) else (
    echo.
    echo ================================================================================
    echo Download interrupted
    echo ================================================================================
    echo.
    echo Progress has been saved. To resume, just run this script again:
    echo   download_model_simple.bat
    echo.
)

pause

