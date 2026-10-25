@echo off
chcp 65001 >nul
cd /d "%~dp0"
set MPLCONFIGDIR=%~dp0.mplcache

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python venv not found in .venv\
    echo.
    echo Run these commands first:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo     .venv\Scripts\python.exe download.py
    echo.
    pause
    exit /b 1
)

if not exist "output" mkdir "output"

rem Pre-flight check: import the app so errors show here instead of a silent exit
".venv\Scripts\python.exe" -c "import gui.app" 2>"output\launch_error.log"
if errorlevel 1 (
    echo [ERROR] Startup check failed:
    echo --------------------------------------------------
    type "output\launch_error.log"
    echo --------------------------------------------------
    pause
    exit /b 1
)
if exist "output\launch_error.log" del "output\launch_error.log" >nul 2>&1

rem launcher.py 是统一入口，和打包后的 exe 走同一条启动路径
start "" ".venv\Scripts\pythonw.exe" "launcher.py"
exit /b 0
