@echo off
setlocal
cd /d "%~dp0.."

set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64

set LOGFILE=test_results.log
echo [%date% %time%] akasha-core test run > "%LOGFILE%"

echo.
echo === [1/3] cargo check --lib --features cuda ===
cargo check --lib --features cuda >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_check
echo OK

echo.
echo === [2/3] cargo test --lib --features cuda -- --test-threads=1 ===
cargo test --lib --features cuda -- --test-threads=1 >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_test
echo OK

echo.
echo === [3/3] cargo build --release --features cuda --bin akasha-core ===
cargo build --release --features cuda --bin akasha-core >> "%LOGFILE%" 2>&1
if errorlevel 1 goto :fail_build
echo OK

echo.
echo ============================================
echo ALL CHECKS PASSED: compile, existing test suite, release build.
echo.
echo IMPORTANT -- read this:
echo This does NOT prove the CUDA-specific kernels (FlashAttention, fused
echo RoPE/QKV, TF32, CUDA Graphs, on-device AdamW schedule, f16 GEMM) are
echo numerically correct end-to-end. The existing tests in step 2 all run
echo on wgpu/Vulkan internally, not CUDA -- that gap existed before today
echo too, nothing new broke it, but it means "tests pass" here only proves
echo the CUDA feature compiles cleanly, not that it computes the right
echo numbers on real hardware.
echo.
echo The only real way to check that is a short manual training run:
echo   1. Run train.bat yourself (or target\release\akasha-core.exe directly)
echo   2. Watch training.log for the first few dozen steps
echo   3. Loss should be a normal finite number and roughly trend down --
echo      NOT "NaN" or "inf"
echo.
echo I did not automate that step here on purpose: if a real training run
echo is already going in another window, a script that starts a second
echo one and then kills "akasha-core.exe" by name could kill the wrong
echo one. That decision needs a human looking at what's actually running,
echo not a script guessing.
echo.
echo Full log: %LOGFILE%
echo ============================================
pause
exit /b 0

:fail_check
echo.
echo FAILED at step 1: cargo check --lib --features cuda
echo Open %LOGFILE% and look at the last ~50 lines for the compiler error.
pause
exit /b 1

:fail_test
echo.
echo FAILED at step 2: cargo test --lib --features cuda
echo Open %LOGFILE% and look at the last ~80 lines.
pause
exit /b 1

:fail_build
echo.
echo FAILED at step 3: cargo build --release --features cuda --bin akasha-core
echo Open %LOGFILE% and look at the last ~80 lines.
pause
exit /b 1
