@echo off
REM Complete pipeline for transcription and WER evaluation
REM Windows Batch Script

echo ========================================
echo Qwen3-Omni Transcription ^& WER Pipeline
echo ========================================
echo.

REM Activate virtual environment (from parent directory)
echo [1/4] Activating virtual environment...
if exist "..\\.venv\Scripts\activate.bat" (
    call ..\\.venv\Scripts\activate.bat
    echo + Virtual environment activated
) else (
    echo X Virtual environment not found. Please run: setup.bat
    exit /b 1
)
echo.

REM Check if dataset exists
echo [2/4] Checking dataset...
if exist "E:\Dataset\LNV\L2-ARCTIC\l2arctic_release_v5.0" (
    for /f %%i in ('dir /b /a:d "E:\Dataset\LNV\L2-ARCTIC\l2arctic_release_v5.0\*" 2^>nul ^| find /c /v ""') do set speaker_count=%%i
    echo + Found dataset with multiple speakers
) else (
    echo X L2-ARCTIC dataset not found at E:\Dataset\LNV\L2-ARCTIC\l2arctic_release_v5.0
    echo   Please place your dataset in the correct directory
    exit /b 1
)
echo.

REM Run transcription
echo [3/4] Running transcription...
echo This may take a while depending on dataset size and hardware...
cd ..
python transcribe.py
if %errorlevel% neq 0 (
    echo X Transcription failed
    cd scripts
    exit /b 1
)
echo + Transcription completed
echo.

REM Run evaluation
echo [4/4] Running WER evaluation...
python evaluate_wer.py
if %errorlevel% neq 0 (
    echo X Evaluation failed
    cd scripts
    exit /b 1
)
echo + Evaluation completed
echo.

echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo.
echo Results saved in .\outputs\
echo   - transcriptions.json
echo   - evaluation_results.json
echo   - evaluation_results.csv
echo.
cd scripts
pause

