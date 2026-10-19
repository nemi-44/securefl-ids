#!/usr/bin/env bash
# Creates a pinned virtual environment and downloads + checksums UNSW-NB15.
set -euo pipefail; cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv && . .venv/bin/activate
pip install --quiet --upgrade pip==24.2 && pip install --quiet -r requirements.txt
mkdir -p data; B=https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification
for f in UNSW_NB15_training-set.csv UNSW_NB15_testing-set.csv; do [ -f data/$f ] || curl -sfL -o data/$f "$B/$f"; done
sha256sum -c data/SHA256SUMS
echo "setup ok"
