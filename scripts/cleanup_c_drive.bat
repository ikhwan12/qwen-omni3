@echo off
echo ================================================================================
echo Clean Up C: Drive Space
echo ================================================================================
echo.
echo WARNING: This will delete cached files. Make sure you don't need them!
echo.
echo Current C: drive space:
dir C:\ /-c | find "bytes free"
echo.
echo.
echo Select what to clean:
echo   1. Hugging Face cache (models, datasets)
echo   2. Python cache (pip, conda)
echo   3. Windows temp files
echo   4. All of the above
echo   5. Show what can be deleted (safe mode)
echo.
set /p choice="Enter choice (1-5): "

if "%choice%"=="1" goto clean_hf
if "%choice%"=="2" goto clean_python
if "%choice%"=="3" goto clean_temp
if "%choice%"=="4" goto clean_all
if "%choice%"=="5" goto show_files
goto end

:clean_hf
echo.
echo Cleaning Hugging Face cache...
if exist "%USERPROFILE%\.cache\huggingface" (
    echo Found: %USERPROFILE%\.cache\huggingface
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\.cache\huggingface" /s /-c ^| find "bytes"') do echo Size: %%a bytes
    echo.
    set /p confirm="Delete Hugging Face cache? (y/N): "
    if /i "%confirm%"=="y" (
        rmdir /s /q "%USERPROFILE%\.cache\huggingface"
        echo Hugging Face cache deleted.
    ) else (
        echo Cancelled.
    )
) else (
    echo No Hugging Face cache found.
)
goto end

:clean_python
echo.
echo Cleaning Python cache...
if exist "%USERPROFILE%\AppData\Local\pip\cache" (
    echo Found: %USERPROFILE%\AppData\Local\pip\cache
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\AppData\Local\pip\cache" /s /-c ^| find "bytes"') do echo Size: %%a bytes
    echo.
    set /p confirm="Delete pip cache? (y/N): "
    if /i "%confirm%"=="y" (
        rmdir /s /q "%USERPROFILE%\AppData\Local\pip\cache"
        echo pip cache deleted.
    ) else (
        echo Cancelled.
    )
) else (
    echo No pip cache found.
)

if exist "%USERPROFILE%\.conda\pkgs" (
    echo Found: %USERPROFILE%\.conda\pkgs
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\.conda\pkgs" /s /-c ^| find "bytes"') do echo Size: %%a bytes
    echo.
    set /p confirm="Delete conda cache? (y/N): "
    if /i "%confirm%"=="y" (
        rmdir /s /q "%USERPROFILE%\.conda\pkgs"
        echo conda cache deleted.
    ) else (
        echo Cancelled.
    )
) else (
    echo No conda cache found.
)
goto end

:clean_temp
echo.
echo Cleaning Windows temp files...
echo Cleaning %TEMP%...
for /f "tokens=3" %%a in ('dir "%TEMP%" /s /-c ^| find "bytes"') do echo Size: %%a bytes
set /p confirm="Delete temp files? (y/N): "
if /i "%confirm%"=="y" (
    del /q /f /s "%TEMP%\*" 2>nul
    echo Temp files deleted.
) else (
    echo Cancelled.
)

echo Cleaning Windows\Temp...
for /f "tokens=3" %%a in ('dir "C:\Windows\Temp" /s /-c ^| find "bytes"') do echo Size: %%a bytes
set /p confirm="Delete Windows temp files? (y/N): "
if /i "%confirm%"=="y" (
    del /q /f /s "C:\Windows\Temp\*" 2>nul
    echo Windows temp files deleted.
) else (
    echo Cancelled.
)
goto end

:clean_all
echo.
echo Cleaning all caches...
call :clean_hf
call :clean_python
call :clean_temp
goto end

:show_files
echo.
echo ================================================================================
echo Files that can be safely deleted:
echo ================================================================================
echo.

echo 1. Hugging Face cache:
if exist "%USERPROFILE%\.cache\huggingface" (
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\.cache\huggingface" /s /-c ^| find "bytes"') do echo   Size: %%a bytes
    echo   Location: %USERPROFILE%\.cache\huggingface
) else (
    echo   Not found
)

echo.
echo 2. pip cache:
if exist "%USERPROFILE%\AppData\Local\pip\cache" (
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\AppData\Local\pip\cache" /s /-c ^| find "bytes"') do echo   Size: %%a bytes
    echo   Location: %USERPROFILE%\AppData\Local\pip\cache
) else (
    echo   Not found
)

echo.
echo 3. conda cache:
if exist "%USERPROFILE%\.conda\pkgs" (
    for /f "tokens=3" %%a in ('dir "%USERPROFILE%\.conda\pkgs" /s /-c ^| find "bytes"') do echo   Size: %%a bytes
    echo   Location: %USERPROFILE%\.conda\pkgs
) else (
    echo   Not found
)

echo.
echo 4. Temp files:
for /f "tokens=3" %%a in ('dir "%TEMP%" /s /-c ^| find "bytes"') do echo   User temp: %%a bytes
for /f "tokens=3" %%a in ('dir "C:\Windows\Temp" /s /-c ^| find "bytes"') do echo   Windows temp: %%a bytes

echo.
echo 5. Other large folders to check:
echo   - Downloads folder: %USERPROFILE%\Downloads
echo   - Recycle Bin: Check if it's full
echo   - Browser caches (Chrome, Firefox, Edge)
echo.
goto end

:end
echo.
echo Current C: drive space after cleanup:
dir C:\ /-c | find "bytes free"
echo.
echo Done!
pause
