@echo off
cd /d "%~dp0.."
set PATH=%PATH%;%USERPROFILE%\.cargo\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64

:loop
echo [%date% %time%] Starting/resuming training... >> training.log
cargo run --release --features cuda --bin akasha-core >> training.log 2>&1
echo [%date% %time%] Process exited -- restarting in 10 seconds... >> training.log
timeout /t 10
goto loop
