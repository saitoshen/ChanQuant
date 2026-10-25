@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "PYW=%ROOT%.venv\Scripts\pythonw.exe"
set "PRELOG=%ROOT%output\launch_error.log"
set "MPLCONFIGDIR=%ROOT%.mplcache"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "ERRLOG=%LOCALAPPDATA%\TrendDesk\logs\error.log"

if not exist "%PY%" goto no_venv
if not exist "%ROOT%output" mkdir "%ROOT%output"

rem Step 1: import check. A broken environment fails here, not silently later.
"%PY%" -c "import gui.app" 2>"%PRELOG%"
if errorlevel 1 goto bad_import
del "%PRELOG%" >nul 2>&1

rem Step 2: archive any previous crash log so the check below sees only this run.
if exist "%ERRLOG%" move /y "%ERRLOG%" "%ERRLOG%.prev" >nul 2>&1

rem Step 3: launch detached (pythonw has no console). If the app dies on startup
rem it writes a traceback to error.log, which we pick up after a short wait.
echo Starting TrendDesk, please wait...
start "" "%PYW%" "%ROOT%launcher.py"
ping -n 5 127.0.0.1 >nul 2>&1
if not exist "%ERRLOG%" exit /b 0

rem Step 4: it crashed. Show the traceback, then retry in console mode.
chcp 65001 >nul
echo.
echo [ERROR] TrendDesk crashed during startup. Traceback:
echo ----------------------------------------------------------------------
type "%ERRLOG%"
echo ----------------------------------------------------------------------
echo.
echo Retrying in console mode so you can see what happens...
echo.
"%PY%" "%ROOT%launcher.py"
echo.
pause
exit /b 1

:no_venv
echo.
echo [ERROR] Python virtualenv not found:
echo     %PY%
echo.
echo Create it first:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1

:bad_import
chcp 65001 >nul
echo.
echo [ERROR] The app failed to import. Details:
echo ----------------------------------------------------------------------
type "%PRELOG%"
echo ----------------------------------------------------------------------
echo.
pause
exit /b 1
