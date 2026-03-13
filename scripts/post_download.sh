#!/bin/bash
# After curl downloads complete, link files into HF cache and split weights.
set -e

DOWNLOAD_DIR="/home/elnur/gpt-oss-offload/downloads"
SNAPSHOT_DIR="/home/elnur/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
OUTPUT_DIR="/home/elnur/gpt-oss-offload/weights"

echo "Linking downloaded shards into HF snapshot directory..."
for f in "$DOWNLOAD_DIR"/model-*.safetensors; do
    fname=$(basename "$f")
    target="$SNAPSHOT_DIR/$fname"
    if [ ! -e "$target" ]; then
        ln -s "$f" "$target"
        echo "  Linked $fname"
    else
        echo "  $fname already exists"
    fi
done

echo ""
echo "Verifying all shards present..."
for i in $(seq 0 14); do
    fname=$(printf "model-%05d-of-00014.safetensors" $i)
    if [ -e "$SNAPSHOT_DIR/$fname" ]; then
        echo "  OK: $fname"
    else
        echo "  MISSING: $fname"
        exit 1
    fi
done

echo ""
echo "Splitting weights..."
cd /home/elnur/gpt-oss-offload
python3 -m scripts.split_weights --model-dir "$SNAPSHOT_DIR" --output-dir "$OUTPUT_DIR"

echo ""
echo "Done! Weights in $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
