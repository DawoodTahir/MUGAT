#!/bin/bash
#SBATCH --job-name=airsim-magnet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR/.."
cd "$PROJECT_ROOT"

if [[ -z "${CONDA_EXE:-}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
conda activate pet

export RAY_TMPDIR="$PWD/.ray_tmp"
mkdir -p "$RAY_TMPDIR"

export AIRSIM_IP=${AIRSIM_IP:-127.0.0.1}
export NUM_WORKERS=${NUM_WORKERS:-1}
export TRAIN_SEED=${TRAIN_SEED:-42}

echo "Running MAGNET training on $(hostname) with seed=$TRAIN_SEED"
python -u train_magnet.py

