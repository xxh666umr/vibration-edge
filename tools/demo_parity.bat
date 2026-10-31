@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONPATH=%~dp0..\.deps"
set "PY=C:\Users\lenovo\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" (
  echo [FAIL] bundled python not found: %PY%
  pause
  exit /b 1
)
"%PY%" tools\test_parity_vib3.py --model-dir models\relabel_xyz_t90 --ckpt models\relabel_xyz_t90\relabel_xyz.pt
echo.
pause
