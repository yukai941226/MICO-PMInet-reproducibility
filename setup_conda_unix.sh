#!/usr/bin/env bash
set -euo pipefail

# Run this file from a shell on a new macOS or Linux computer.
cd "$(dirname "$0")"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] Conda was not found. Install Miniconda/Anaconda and try again." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

echo "[1/4] Creating the mico-pminet Conda environment..."
conda env create -f environment.yml

echo "[2/4] Activating the environment and installing this repository..."
conda activate mico-pminet
python -m pip install --no-deps -e .

echo "[3/4] Running automated tests..."
python -m pytest -q

echo "[4/4] Verifying released data, checkpoint, and numerical results..."
python run.py verify

echo "[SUCCESS] The reproducibility environment is ready."
echo "For later sessions: conda activate mico-pminet"
