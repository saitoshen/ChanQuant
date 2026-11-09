@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  ChanQuant launcher (source checkout)
rem
rem  Interpreter lookup, first hit wins:
rem    1. .venv\Scripts\python.exe        project venv (canonical dev env)
rem    2. %CHANQUANT_PY%\python.exe       explicit override, point at any env
rem    3. common conda / anaconda roots
rem    4. python.exe on PATH (last resort)
rem
rem  This used to require .venv and nothing else: with no venv it printed
rem  "Python virtualenv not found" and exited. On a machine that has a working
rem  conda env that is a dead end for no reason, so a missing venv now falls
rem  through to the other candidates instead of failing.
rem
rem  IMPORTANT: keep this file pure ASCII with CRLF endings. cmd.exe reads .bat
rem  as the OEM codepage; a stray non-ASCII byte (even inside rem) can be split
rem  into garbage commands. LF-only ends break labels and goto.
rem ---------------------------------------------------------------------------

set "ROOT=%~dp0"
set "PY="
set "PYW="
set "PRELOG=%ROOT%output\launch_error.log"
set "MPLCONFIGDIR=%ROOT%.mplcache"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"
set "ERRLOG=%LOCALAPPDATA%\ChanQuant\logs\error.log"

if exist "%ROOT%.venv\Scripts\python.exe" call :use "%ROOT%.venv\Scripts\python.exe"
if not defined PY if defined CHANQUANT_PY call :use "%CHANQUANT_PY%\python.exe"
if not defined PY if exist "%ProgramData%\anaconda3\python.exe" call :use "%ProgramData%\anaconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\python.exe" call :use "%USERPROFILE%\anaconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\anaconda3\python.exe" call :use "%LOCALAPPDATA%\anaconda3\python.exe"
if not defined PY if exist "%ProgramData%\miniconda3\python.exe" call :use "%ProgramData%\miniconda3\python.exe"
if not defined PY if exist "%USERPROFILE%\miniconda3\python.exe" call :use "%USERPROFILE%\miniconda3\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\miniconda3\python.exe" call :use "%LOCALAPPDATA%\miniconda3\python.exe"
if not defined PY for /f "delims=" %%I in ('where python 2^>nul') do if not defined PY call :use "%%I"

if not defined PY goto no_python
if not exist "%ROOT%output" mkdir "%ROOT%output"

rem Step 1: import check. A broken environment fails here, not silently later.
echo Using python: %PY%
"%PY%" -c "import gui.app" 2>"%PRELOG%"
if errorlevel 1 goto bad_import
del "%PRELOG%" >nul 2>&1

rem Step 2: archive any previous crash log so the check below sees only this run.
if exist "%ERRLOG%" move /y "%ERRLOG%" "%ERRLOG%.prev" >nul 2>&1

rem Step 3: launch detached (pythonw has no console). If the app dies on startup
rem it writes a traceback to error.log, which we pick up after a short wait.
echo Starting ChanQuant, please wait...
start "" "%PYW%" "%ROOT%launcher.py"
ping -n 5 127.0.0.1 >nul 2>&1
if not exist "%ERRLOG%" exit /b 0

rem Step 4: it crashed. Show the traceback, then retry in console mode.
chcp 65001 >nul
echo.
echo [ERROR] ChanQuant crashed during startup. Traceback:
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

:use
rem %~1 = python.exe. Derive pythonw.exe next to it; fall back to python.exe
rem when this env has no windowless build.
if not exist "%~1" goto :eof
set "PY=%~1"
set "PYW=%~dp1pythonw.exe"
if not exist "%PYW%" set "PYW=%PY%"
goto :eof

:no_python
echo.
echo [ERROR] No usable Python found.
echo.
echo Looked for, in order:
echo     %ROOT%.venv\Scripts\python.exe
echo     %%CHANQUANT_PY%%\python.exe
echo     anaconda3 or miniconda3 under ProgramData, USERPROFILE, LOCALAPPDATA
echo     python.exe on PATH
echo.
echo Fix it either way:
echo     A) create the project venv:
echo            python -m venv .venv
echo            .venv\Scripts\python.exe -m pip install -r requirements.txt
echo     B) point at an env that already has the dependencies:
echo            set CHANQUANT_PY=C:\ProgramData\anaconda3
echo.
pause
exit /b 1

:bad_import
chcp 65001 >nul
echo.
echo [ERROR] The app failed to import with this interpreter:
echo     %PY%
echo.
echo Details:
echo ----------------------------------------------------------------------
type "%PRELOG%"
echo ----------------------------------------------------------------------
echo.
echo Most likely the dependencies are missing in that environment:
echo     "%PY%" -m pip install -r requirements.txt
echo.
pause
exit /b 1
