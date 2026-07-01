#!/usr/bin/env bash
cd "$(dirname "$0")/.."

export PATH="$PATH:$HOME/.cargo/bin:/usr/local/cuda/bin"

while true; do
    echo "[$(date)] Starting/resuming training..." >>training.log
    cargo run --release --features cuda --bin akasha-core >>training.log 2>&1
    echo "[$(date)] Process exited -- restarting in 10 seconds..." >>training.log
    sleep 10
done
