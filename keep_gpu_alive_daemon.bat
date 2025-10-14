@echo off
REM Launcher for GPU keep-alive daemon script
cd scripts_linux
call keep_gpu_alive_daemon.sh %*
cd ..
