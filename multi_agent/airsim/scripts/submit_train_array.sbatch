#!/bin/bash
#SBATCH --job-name=airsim-ma-ppo-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4
#SBATCH --output=results/%x_%A_%a.o
#SBATCH --error=results/%x_%A_%a.e

set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR/.."
cd "$PROJECT_ROOT"

# Ensure results directory exists for SLURM logs
mkdir -p results

# Optional: Enable site module environment if available (HPC vendor-specific)
if [[ -f "/opt/flight/etc/setup.sh" ]]; then
    source "/opt/flight/etc/setup.sh"
    flight env activate gridware || true
fi

# Optional: Use environment modules if present (safe no-ops otherwise)
if command -v module &>/dev/null; then
    module purge || true
    # Load cluster-provided Python/CUDA if desired; keep best-effort with guards
    module load python/3.11 || true
    module load libs/nvidia-cuda/11.8.0/bin || true
fi

# Activate conda env
if [[ -z "${CONDA_EXE:-}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Create or update the environment from the cluster spec, then activate it
ENV_NAME="pet"
ENV_FILE="$PROJECT_ROOT/../environment.cluster.yml"
REQ_FILE="$PROJECT_ROOT/../requirements-train.txt"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Conda env '$ENV_NAME' not found. Creating from $ENV_FILE ..."
    conda env create -f "$ENV_FILE"
else
    echo "Conda env '$ENV_NAME' exists. Updating from $ENV_FILE ..."
    conda env update -f "$ENV_FILE" --prune
fi

conda activate "$ENV_NAME"

# Ensure pip requirements are up to date inside the env (safe if already installed)
python -m pip install --upgrade pip
python -m pip install -r "$REQ_FILE"

export RAY_TMPDIR="$PWD/.ray_tmp"
mkdir -p "$RAY_TMPDIR"

export AIRSIM_IP=${AIRSIM_IP:-86.26.201.86}
export NUM_WORKERS=${NUM_WORKERS:-1}

# Derive a seed from the array index
export TRAIN_SEED=$(( 1000 + SLURM_ARRAY_TASK_ID ))

echo "Launching array task $SLURM_ARRAY_TASK_ID with seed=$TRAIN_SEED"
python -u train.py

