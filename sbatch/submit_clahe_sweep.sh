#!/bin/bash
# ==============================================================================
# submit_clahe_sweep.sh - Submit CLAHE + Unsharp Masking Parameter Sweep to Slurm
# Evaluates different spatial scales (sigma) and edge boost intensities (gamma).
# ==============================================================================
# Usage:
#   bash sbatch/submit_clahe_sweep.sh
# ==============================================================================

set -e

echo "=========================================================="
echo "Submitting CLAHE + Unsharp Mask Parameter Sweep Jobs"
echo "Target: Evaluate (sigma, gamma) combinations for SMT micro-capacitors"
echo "=========================================================="

# Combination 1: Fine-scale subtle sharpening (targets 0402 micro pads without grain explosion)
sbatch sbatch/train_clahe_unsharp.sh 0.5 0.3 1.5

# Combination 2: Standard-scale subtle sharpening
sbatch sbatch/train_clahe_unsharp.sh 1.0 0.3 1.5

# Combination 3: Standard-scale moderate sharpening (trained counterpart of Exp 4)
sbatch sbatch/train_clahe_unsharp.sh 1.0 0.6 1.5

# Combination 4: Structural wide-scale sharpening (sharpens component contours, suppresses grain)
sbatch sbatch/train_clahe_unsharp.sh 1.5 0.4 1.5

echo ""
echo "✅ All 4 parameter sweep jobs submitted to Slurm!"
echo "Check queue status with: squeue -u etadevosyan"
