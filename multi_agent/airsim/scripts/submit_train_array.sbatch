#!/bin/bash
#SBATCH --job-name=airsim-ma-ppo-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR/.."
cd "$PROJECT_ROOT"

# Activate conda env
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

# Derive a seed from the array index
export TRAIN_SEED=$(( 1000 + SLURM_ARRAY_TASK_ID ))

echo "Launching array task $SLURM_ARRAY_TASK_ID with seed=$TRAIN_SEED"
python -u train.py

