#!/bin/bash
#SBATCH --job-name=airsim-ma-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Optional: adjust these paths for your cluster
PROJECT_ROOT="$SLURM_SUBMIT_DIR/.."   # points to multi_agent/airsim
cd "$PROJECT_ROOT"

# Load modules if your cluster uses Lmod (uncomment and adjust)
# module purge
# module load cuda/11.7

# Activate conda env
if command -v module &>/dev/null; then
    :
fi

if [[ -z "${CONDA_EXE:-}" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
fi
conda activate pet

# Ray sometimes needs a temp dir writable on shared FS
export RAY_TMPDIR="$PWD/.ray_tmp"
mkdir -p "$RAY_TMPDIR"

# Configure AirSim IP (e.g., simulator node) and RLlib params
export AIRSIM_IP=${AIRSIM_IP:-127.0.0.1}
export NUM_WORKERS=${NUM_WORKERS:-1}
export TRAIN_SEED=${TRAIN_SEED:-42}

# Optional resume: pass absolute path to rllib_checkpoint.json
# export RESUME_FROM="/path/to/saved_policy/final_xxx/rllib_checkpoint.json"

echo "Running on host $(hostname)"
echo "AIRSIM_IP=$AIRSIM_IP NUM_WORKERS=$NUM_WORKERS TRAIN_SEED=$TRAIN_SEED"

python -u train.py

