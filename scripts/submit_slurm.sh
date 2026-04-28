#!/bin/bash
# SLURM job script for FMCIB feature extraction — Humanitas meningioma cohort.
#
# BEFORE SUBMITTING:
#   1. Set PROJECT_DIR to the absolute path of this repo on the HPC.
#   2. Adjust --partition, --time, --mem, --gres as needed.
#   3. Edit configs/humanitas_train_flat.yaml: fill in volumes_root and
#      segmentations_root with the real data paths on this HPC.
#   4. Make sure model_weights.torch is present in PROJECT_DIR (scp it from
#      your laptop if the compute nodes lack internet access).
#
# TEST RUN FIRST (5 patients, no queue delay):
#   bash scripts/submit_slurm.sh --test
#
# FULL RUN:
#   sbatch scripts/submit_slurm.sh

# ---------------------------------------------------------------------------
# SLURM directives — adjust to your HPC
# ---------------------------------------------------------------------------
#SBATCH --job-name=fmcib_humanitas
#SBATCH --output=logs/fmcib_%j.out
#SBATCH --error=logs/fmcib_%j.err
#SBATCH --time=02:00:00        # 110 patients x 4 modalities ≈ 30–60 min on GPU
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1           # remove line if no GPU available
#SBATCH --partition=gpu        # ← edit to match HPC partition name

# ---------------------------------------------------------------------------
# EDIT THIS
# ---------------------------------------------------------------------------
PROJECT_DIR=/path/to/dl_pipeline    # ← absolute path to repo on HPC

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
set -euo pipefail

# Uncomment and adjust module loads to match the HPC environment:
# module load python/3.12
# module load cuda/12.1

cd "$PROJECT_DIR"
mkdir -p logs

source .venv/bin/activate

CONFIG=configs/humanitas_train_flat.yaml

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--test" ]]; then
    echo "=== TEST RUN: 5 patients ==="
    python scripts/extract_features.py --config "$CONFIG" --max-patients 5
else
    echo "=== FULL RUN ==="
    python scripts/extract_features.py --config "$CONFIG"
fi
