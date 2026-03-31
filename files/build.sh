#!/bin/bash
set -e
echo "Building JupyterLite..."
jupyter lite build --contents .
echo "Copying assets..."
mkdir -p _output/files/trainer
mkdir -p _output/data
cp files/trainer/engine.js _output/files/trainer/engine.js
cp files/data/train_chunks.json _output/data/train_chunks.json
cp files/data/tokenizer.json _output/data/tokenizer.json
echo "Done. Serve with: cd _output && python3 -m http.server 8080"
