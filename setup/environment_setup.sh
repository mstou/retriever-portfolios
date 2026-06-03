# Make sure tools use your PVC as HOME (avoids the uid/passwd issue)
export HOME="${HOME:-$(pwd)}"
mkdir -p $HOME/.local/bin

# If pip is missing, bootstrap it (safe to re-run)
python3 -m pip --version 2>/dev/null || \
( curl -sS https://bootstrap.pypa.io/get-pip.py -o $HOME/get-pip.py && python3 $HOME/get-pip.py --user )

# Install virtualenv for your user (no sudo)
python3 -m pip install --user --upgrade virtualenv

# Add ~/.local/bin to PATH for this session
export PATH=$HOME/.local/bin:$PATH

# Create a persistent env on your PVC
virtualenv -p python3 "${VENV_DIR:-.venv}"

# Activate it
source "${VENV_DIR:-.venv}/bin/activate"

# Install your libraries once
pip install --upgrade pip
pip install -r requirements.txt
