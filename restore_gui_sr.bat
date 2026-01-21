@echo off
setlocal EnableExtensions

REM ====== Config ======
set "REPO_DIR=D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100"
set "TARGET_FILE=(gui)super-resolution processing.py"
set "REMOTE_NAME=origin"
set "REMOTE_BRANCH="
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

git remote get-url "%REMOTE_NAME%" >nul 2>&1 || (
  echo [ERROR] Remote not found: "%REMOTE_NAME%"
  pause
  exit /b 1
)

echo [TIP] Repo directory: "%REPO_DIR%"
echo [INFO] Current status (before restore):
git status -sb
echo.

echo [INFO] Fetching...
git fetch "%REMOTE_NAME%" || (
  echo [ERROR] git fetch failed.
  pause
  exit /b 1
)

if not defined REMOTE_BRANCH (
  for /f "delims=" %%b in ('git symbolic-ref -q --short refs/remotes/%REMOTE_NAME%/HEAD 2^>nul') do set "REMOTE_BRANCH=%%b"
)

if not defined REMOTE_BRANCH (
  git show-ref --verify --quiet refs/remotes/%REMOTE_NAME%/main
  if not errorlevel 1 set "REMOTE_BRANCH=%REMOTE_NAME%/main"
)

if not defined REMOTE_BRANCH (
  git show-ref --verify --quiet refs/remotes/%REMOTE_NAME%/master
  if not errorlevel 1 set "REMOTE_BRANCH=%REMOTE_NAME%/master"
)

if not defined REMOTE_BRANCH (
  echo [ERROR] Cannot find remote branch on "%REMOTE_NAME%".
  echo Tip: set REMOTE_BRANCH manually or ensure remote has main/master.
  pause
  exit /b 1
)

echo [INFO] Restoring "%TARGET_FILE%" from "%REMOTE_BRANCH%"
git checkout "%REMOTE_BRANCH%" -- "%TARGET_FILE%" || (
  echo [ERROR] git checkout restore failed.
  echo Tip: make sure branch exists on "%REMOTE_NAME%".
  pause
  exit /b 1
)

echo.
echo [INFO] Current status (after restore):
git status -sb

echo [DONE] Restore completed.
pause
exit /b 0