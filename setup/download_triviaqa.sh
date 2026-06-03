TRIVIAQA_DIR="${TRIVIAQA_DIR:-datasets/TriviaQA}"

mkdir -p "$TRIVIAQA_DIR"

# Download DPR TriviaQA train/dev JSON (gzipped)
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz -P "$TRIVIAQA_DIR"
wget https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz -P "$TRIVIAQA_DIR"

# Unzip in place
gunzip -f "$TRIVIAQA_DIR/biencoder-trivia-train.json.gz"
gunzip -f "$TRIVIAQA_DIR/biencoder-trivia-dev.json.gz"

# Rename to standard train/test filenames
mv "$TRIVIAQA_DIR/biencoder-trivia-train.json" "$TRIVIAQA_DIR/train.json"
mv "$TRIVIAQA_DIR/biencoder-trivia-dev.json" "$TRIVIAQA_DIR/test.json"
