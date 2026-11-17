#!/bin/bash
# Tests + benchmarks on one GPU. Submit from the repo root: sbatch benchmarks/slurm.sh
# Adjust partition/QOS/modules to your cluster (copy them from a job script you already use).
#SBATCH --job-name=so3kit-bench
#SBATCH --partition=gpu             # change to your GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=so3kit-bench-%j.log

set -e
# module load CUDA/12.x             # if your cluster uses modules
source .venv/bin/activate           # created once: python3.12 -m venv .venv && pip install -e ".[test]" -r benchmarks/requirements.txt

nvidia-smi
pytest -v
python examples/maximize.py
bash benchmarks/run_all.sh
