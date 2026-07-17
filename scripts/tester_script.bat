@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64

set LOGFILE=test_results.log
echo [%date% %time%] akasha-core test run > "%LOGFILE%"

set ANY_FAILED=0

echo === [1/2] cargo test --lib --features cuda -- --test-threads=1 ===
cargo test --lib --features cuda -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S1=FAILED& set ANY_FAILED=1& echo FAILED) else (set S1=OK& echo OK)

echo === [2/2] cargo build --release --features cuda --bin akasha-core ===
cargo build --release --features cuda --bin akasha-core >> "%LOGFILE%" 2>&1
if errorlevel 1 (set S2=FAILED& set ANY_FAILED=1& echo FAILED) else (set S2=OK& echo OK)

echo.
echo ============================================
echo   [1/2] test  --lib --features cuda ....... !S1!
echo   [2/2] build --release --features cuda ... !S2!
echo ============================================

if "!ANY_FAILED!"=="1" (
    echo FAILED -- send back the whole %LOGFILE%
    pause
    exit /b 1
)

echo ALL PASSED. Log: %LOGFILE%
pause
exit /b 0
