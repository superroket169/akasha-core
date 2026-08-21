#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export PATH="$PATH:$HOME/.cargo/bin"

SHARD="${1:-0000}"
URL="https://huggingface.co/datasets/Skylion007/openwebtext/resolve/refs%2Fconvert%2Fparquet/plain_text/train/${SHARD}.parquet"
PARQUET="data/openwebtext_${SHARD}.parquet"

mkdir -p data
echo "Downloading $URL"
curl -L --fail -o "$PARQUET" "$URL"

echo "Building prepare_dataset"
cargo build --release --bin prepare_dataset

echo "Converting to data/train.txt + data/eval.txt"
./target/release/prepare_dataset "$PARQUET"

rm -f "$PARQUET"

# New corpus -- old token shards are for the old train.txt.
echo "Clearing stale token shards"
rm -rf data/train_shards

echo "Done. data/train.txt + data/eval.txt ready."
