#!/usr/bin/env bash
# Run every benchmark and redraw the README figure. Usage: bash benchmarks/run_all.sh
set -e
cd "$(dirname "$0")"
python bench_wigner.py
python bench_sofft.py
python bench_derivatives.py
python plot.py
