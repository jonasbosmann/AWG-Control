@echo off
REM Shortcut for the most common pipeline_view.py invocation: the 8-panel
REM overview dashboard. Works from any directory, and double-clickable from
REM Explorer -- both cd's to its own location first (no need to remember to
REM be inside chirp_bench) and uses the conda env's python.exe by full path
REM (double-clicking a .bat opens a fresh shell that does NOT have a conda
REM env's python on PATH unless the env is activated, which only happens in
REM a terminal you've activated it in yourself -- this sidesteps that).
REM
REM Usage:
REM   overview                  interactive picker
REM   overview 1p70_1p80        match by substring (same as pipeline_view.py)
cd /d "%~dp0"
set PY=C:\Users\Admin\anaconda3\envs\awg\python.exe
if not exist "%PY%" set PY=python
"%PY%" pipeline_view.py %* --metric overview
echo.
echo done -- press any key to close this window
pause >nul
