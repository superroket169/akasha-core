#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export PATH="$PATH:$HOME/.cargo/bin"

SHARD="${1:-000_00000}"
URL="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/${SHARD}.parquet"
PARQUET="data/fineweb_${SHARD}.parquet"

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
