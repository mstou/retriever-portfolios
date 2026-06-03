# Create target directories
TWOWIKI_DIR="${TWOWIKI_DIR:-datasets/2WikiMultiHopQA}"

mkdir -p "$TWOWIKI_DIR"
mkdir -p .temp

# Download 2WikiMultiHopQA zip
wget "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=0" -O .temp/2wikimultihopqa.zip

# Unzip into the dataset directory, flattening paths and ignoring macOS cruft
unzip -jo .temp/2wikimultihopqa.zip -d "$TWOWIKI_DIR" -x "*.DS_Store"

# clean up the temp zip
rm -f .temp/2wikimultihopqa.zip

# We do not use test.json (no supporting facts).
rm -f "$TWOWIKI_DIR/test.json"
