#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

export PATH="$PATH:$HOME/.cargo/bin"

URL="https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
PARQUET="data/dolly-15k.parquet"

mkdir -p data
echo "Downloading $URL"
curl -L --fail -o "$PARQUET" "$URL"

echo "Building prepare_finetune_dataset"
cargo build --release --bin prepare_finetune_dataset

echo "Converting to data/train.txt + data/eval.txt (User:/Assistant: blocks)"
./target/release/prepare_finetune_dataset "$PARQUET"

rm -f "$PARQUET"

# New corpus -- old token shards are for the old train.txt.
echo "Clearing stale token shards"
rm -rf data/train_shards

echo "Done. data/train.txt + data/eval.txt ready (chat finetune format)."
