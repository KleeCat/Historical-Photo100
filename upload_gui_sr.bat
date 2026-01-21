@echo off
setlocal EnableExtensions

REM ====== Config ======
set "REPO_DIR=D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100"
set "TARGET_FILE=(gui)super-resolution processing.py"
set "DEFAULT_MSG=Update GUI script"
set "REMOTE_NAME=origin"
REM ====================

cd /d "%REPO_DIR%" || (
  echo [ERROR] Repo folder not found: "%REPO_DIR%"
  pause
  exit /b 1
)

git rev-parse --is-inside-work-tree >nul 2>&1 || (
  echo [ERROR] Not a git repository: "%REPO_DIR%"
  pause
  exit /b 1
)

echo [TIP] Repo directory: "%REPO_DIR%"
echo [INFO] Current status:
git status -sb
echo.

REM Allow custom commit message: upload_gui_sr.bat "your message"
set "COMMIT_MSG=%~1"
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=%DEFAULT_MSG%"

echo [INFO] Adding file: "%TARGET_FILE%"
git add -- "%TARGET_FILE%" || (
  echo [ERROR] git add failed.
  pause
  exit /b 1
)

REM Exit if no staged changes
git diff --cached --quiet
if %errorlevel%==0 (
  echo [OK] No staged changes. Nothing to commit.
  pause
  exit /b 0
)

echo [INFO] Commit: "%COMMIT_MSG%"
git commit -m "%COMMIT_MSG%" || (
  echo [ERROR] git commit failed.
  pause
  exit /b 1
)

for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%b"
if "%CURRENT_BRANCH%"=="HEAD" (
  echo [ERROR] Detached HEAD. Checkout a branch first.
  pause
  exit /b 1
)

set "UPSTREAM="
for /f "delims=" %%b in ('git rev-parse --abbrev-ref --symbolic-full-name @{u} 2^>nul') do set "UPSTREAM=%%b"

if defined UPSTREAM (
  for /f "tokens=1,* delims=/" %%r in ("%UPSTREAM%") do (
    set "PUSH_REMOTE=%%r"
    set "PUSH_BRANCH=%%s"
  )
) else (
  set "PUSH_REMOTE=%REMOTE_NAME%"
  set "PUSH_BRANCH=%CURRENT_BRANCH%"
)

git remote get-url "%PUSH_REMOTE%" >nul 2>&1 || (
  echo [ERROR] Remote not found: "%PUSH_REMOTE%"
  pause
  exit /b 1
)

echo [INFO] Pushing to "%PUSH_REMOTE%" "%CURRENT_BRANCH%:%PUSH_BRANCH%"
git push "%PUSH_REMOTE%" "%CURRENT_BRANCH%:%PUSH_BRANCH%" || (
  echo [ERROR] git push failed.
  pause
  exit /b 1
)

echo.
echo [INFO] Status after upload:
git status -sb

echo [DONE] Upload completed.
pause
exit /b 0