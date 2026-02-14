@echo off
echo ========================================
echo   Claude Code Environment Setup
echo ========================================
echo.

set /p API_KEY=Please enter your API Key:

if "%API_KEY%"=="" (
    echo Error: API Key cannot be empty!
    pause
    exit /b 1
)

echo.
echo Setting environment variables...

setx ANTHROPIC_AUTH_TOKEN "%API_KEY%"
setx ANTHROPIC_BASE_URL "https://www.fucheers.top"
setx CLAUDE_CODE_MAX_OUTPUT_TOKENS "12000"

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Environment variables set:
echo   ANTHROPIC_AUTH_TOKEN = %API_KEY%
echo   ANTHROPIC_BASE_URL = https://www.fucheers.top
echo   CLAUDE_CODE_MAX_OUTPUT_TOKENS = 12000
echo.
echo Please restart your terminal for changes to take effect.
echo.
pause
