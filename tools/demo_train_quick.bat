@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONPATH=%~dp0..\.deps"
set "PY=C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
"%PY%" tools\train_vib3.py --split random --epochs 2 --out models\_demo_train
echo.
pause
