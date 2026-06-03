MUSIQUE_DIR="${MUSIQUE_DIR:-datasets/MuSiQue}"

mkdir -p "$MUSIQUE_DIR"
wget https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_train.jsonl -P "$MUSIQUE_DIR"
mv "$MUSIQUE_DIR/musique_ans_v1.0_train.jsonl" "$MUSIQUE_DIR/train.jsonl"

wget https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl -P "$MUSIQUE_DIR"
mv "$MUSIQUE_DIR/musique_ans_v1.0_dev.jsonl" "$MUSIQUE_DIR/test.jsonl"
