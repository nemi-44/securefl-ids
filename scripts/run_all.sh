#!/usr/bin/env bash
# Runs the whole local matrix in parallel, then the analysis.
set -euo pipefail; cd "$(dirname "$0")/.."
P=${JOBS:-6}; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/matrix.py | while read -r c; do n=$(echo "$c" | awk "{print \$4\"_s\"\$6}"); [ -f "results/local/$n.json" ] || echo "$c"; done | xargs -P "$P" -I{} bash -c "{}" > results/local_run.log 2>&1
python scripts/analyse.py
