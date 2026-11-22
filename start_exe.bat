@echo off
rem Starts the built liveTranslate.exe from dist\ (keys are read from .env above it).
if not exist "%~dp0dist\liveTranslate\liveTranslate.exe" (
  echo dist\liveTranslate\liveTranslate.exe not found - build it first with:
  echo     python -m PyInstaller liveTranslate.spec --clean --noconfirm
  ping -n 6 127.0.0.1 >nul
  exit /b 1
)
cd /d "%~dp0dist\liveTranslate"
start "" "liveTranslate.exe"
echo liveTranslate.exe is starting - look for its icon near the clock.
ping -n 4 127.0.0.1 >nul
