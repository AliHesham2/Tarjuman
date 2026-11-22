@echo off
rem Starts liveTranslate from the source code (uses the GPU, no console window).
rem Logs: %USERPROFILE%\.liveTranslate\logs\live_translate.log
cd /d "%~dp0"
where pythonw >nul 2>nul && (set "PY=pythonw") || (set "PY=python")
start "" %PY% "%~dp0main.py"
echo liveTranslate is starting - look for its icon near the clock.
echo Listening: Ctrl+Shift+L   Show/hide: Ctrl+Shift+T   Settings: Ctrl+Shift+S
ping -n 4 127.0.0.1 >nul
