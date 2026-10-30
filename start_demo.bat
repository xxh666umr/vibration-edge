@echo off
chcp 65001 >nul
title Vibration Edge Demo - STM32F103
cd /d "%~dp0"
set "PY=python"
if not exist "%PY%" if exist "%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" set "PY=%USERPROFILE%\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
echo Starting vibration dashboard at http://127.0.0.1:8766  (close this window to stop)
"%PY%" demo\server.py --http-port 8766
pause
