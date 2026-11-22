@echo off
rem Stops liveTranslate (source version or the exe). Only this project's processes.
set "APP=%~dp0main.py"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$n=0; Get-CimInstance Win32_Process | Where-Object { ($_.Name -match '^pythonw?\.exe$' -and $_.CommandLine -like ('*' + $env:APP + '*')) -or $_.Name -eq 'liveTranslate.exe' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $n++; 'Stopped ' + $_.Name + ' (PID ' + $_.ProcessId + ')' }; if ($n -eq 0) { 'liveTranslate is not running.' }"
ping -n 3 127.0.0.1 >nul
