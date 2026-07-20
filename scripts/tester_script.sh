#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1

export PATH="$PATH:$HOME/.cargo/bin:/usr/local/cuda/bin"

LOGFILE=test_results.log
echo "[$(date)] akasha-core test run" > "$LOGFILE"

ANY_FAILED=0

echo "=== [1/2] cargo test --features cuda -- --test-threads=1 ==="
cargo test --features cuda -- --test-threads=1 >> "$LOGFILE" 2>&1
if [ $? -ne 0 ]; then S1=FAILED; ANY_FAILED=1; echo FAILED; else S1=OK; echo OK; fi

echo "=== [2/2] cargo build --release --features cuda --bin akasha-core ==="
cargo build --release --features cuda --bin akasha-core >> "$LOGFILE" 2>&1
if [ $? -ne 0 ]; then S2=FAILED; ANY_FAILED=1; echo FAILED; else S2=OK; echo OK; fi

echo
echo "============================================"
echo "  [1/2] test  --features cuda ............. $S1"
echo "  [2/2] build --release --features cuda ... $S2"
echo "============================================"

if [ "$ANY_FAILED" = "1" ]; then
    echo "FAILED -- send back the whole $LOGFILE"
    read -n 1 -s -r -p "Press any key to continue..."
    echo
    exit 1
fi

echo "ALL PASSED. Log: $LOGFILE"
read -n 1 -s -r -p "Press any key to continue..."
echo
exit 0
