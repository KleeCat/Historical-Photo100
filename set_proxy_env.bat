@echo off
setlocal EnableExtensions

REM Set proxy environment variables for this session.
set "HTTP_PROXY=http://127.0.0.1:7897"
set "HTTPS_PROXY=http://127.0.0.1:7897"
set "ALL_PROXY=socks5://127.0.0.1:7897"

echo [INFO] Proxy environment variables set.
echo [INFO] Testing OpenAI endpoint (expect 401 Unauthorized)...
curl -I --proxy socks5h://127.0.0.1:7897 https://api.openai.com/v1/models

pause
exit /b 0
