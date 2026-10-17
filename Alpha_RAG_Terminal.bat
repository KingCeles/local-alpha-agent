@echo off
title Local RTX 3060 Financial RAG Terminal
echo =========================================================
echo   Starting Local RTX 3060 Financial RAG Terminal...
echo =========================================================

:: 1. Start WSL backend in a background process
start /B wsl.exe -e bash -lc "~/local-alpha-agent/start_terminal.sh"

:: 2. Wait 4 seconds for Streamlit server to initialize
timeout /t 4 /nobreak >nul

:: 3. Open in Native Standalone Desktop App Mode (Edge)
start msedge --app=http://localhost:8501 --window-size=1440,900

echo.
echo [+] App window launched!
echo [!] Keep this console minimized while using the app.
echo [!] Press ANY KEY here when you are done to stop Streamlit and free RTX 3060 VRAM...
pause >nul

echo [*] Shutting down Streamlit and unloading Ollama from GPU...
wsl.exe -e bash -lc "pkill -f 'streamlit run'; pkill ollama"
echo [+] Clean shutdown complete.
timeout /t 2 >nul
