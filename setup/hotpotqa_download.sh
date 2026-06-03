#!/bin/bash

HOTPOTQA_DIR="${HOTPOTQA_DIR:-datasets/HotpotQA-BEIR}"

mkdir -p "$HOTPOTQA_DIR"
wget https://huggingface.co/datasets/BeIR/hotpotqa/raw/main/corpus.jsonl.gz -P "$HOTPOTQA_DIR"
gunzip -f "$HOTPOTQA_DIR/corpus.jsonl.gz"
wget https://huggingface.co/datasets/BeIR/hotpotqa/resolve/main/queries.jsonl.gz -P "$HOTPOTQA_DIR"
gunzip -f "$HOTPOTQA_DIR/queries.jsonl.gz"
