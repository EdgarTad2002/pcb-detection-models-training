#!/bin/bash
# ==============================================================================
# Master Launcher for PCB Ablation Experiments 1 to 8 (YOLO26s & YOLO12s)
# Optimized for 4-Canonical-Component pcb-filtered-yolov8 Dataset
#
# Experiments:
#   1: Clean Baseline (YOLO26s & YOLO12s)
#   2: Agaian Retinex Data-Level (YOLO26s & YOLO12s)
#   3: Grand Unified Data-Level (YOLO26s & YOLO12s)
#   4: Loss-Only Ablation: Multi-Stage SDice (YOLO26s & YOLO12s)
#   5: In-Network 2D Conv Stem (YOLO26s & YOLO12s)
#   6: In-Network CTDN Attention Stem (YOLO26s & YOLO12s)
#   7: ALL APPROACHES TOGETHER (YOLO26s & YOLO12s)
#   8: Master Super-Ensemble WBF Consensus
#
# Usage:
#   bash sbatch/launch_all_new_experiments.sh prep      # Verifies & sets up 4-class dataset
#   bash sbatch/launch_all_new_experiments.sh 7         # Submits Exp 7 (Top 2 Champions: YOLO26s & YOLO12s All Approaches)
#   bash sbatch/launch_all_new_experiments.sh best      # Submits Top 4 Models (Exp 7 Champions + Exp 1 Baselines)
#   bash sbatch/launch_all_new_experiments.sh 8         # Runs Exp 8 Master Super-Ensemble
#   bash sbatch/launch_all_new_experiments.sh all       # Submits Exp 1-7 (all 14 ablation jobs)
# ==============================================================================

set -e

MODE=${1:-"best"}

echo "=========================================================================="
echo "🎯 PCB AOI ABLATION EXPERIMENTS LAUNCHER (Mode: $MODE)"
echo "   Target Dataset: 4-Component pcb-filtered-yolov8"
echo "=========================================================================="

# Step 0: Ensure dataset is audited and configured for the 4 canonical components
WORKSPACE_DIR=${PCB_YOLO_WORKSPACE:-"/mnt/weka/etadevosyan/pcb-yolo/pcb-yolo26n-gurgen"}
cd "$WORKSPACE_DIR"
export PYTHONPATH="$WORKSPACE_DIR:$PYTHONPATH"

if [ -f "tools/setup_pcb_filtered_4class.py" ]; then
    echo "🔍 Verifying 4-class configuration for pcb-filtered-yolov8..."
    python tools/setup_pcb_filtered_4class.py
fi

if [[ "$MODE" == "prep" || "$MODE" == "setup" ]]; then
    echo "✅ Dataset preparation complete. Exiting without launching training jobs."
    exit 0
fi

submit_job() {
    local script="$1"
    local desc="$2"
    if [ -f "$script" ]; then
        echo -n "Submitting $desc ($script)... "
        local job_output=$(sbatch "$script")
        echo "$job_output"
    else
        echo "⚠️ Script $script not found, skipping."
    fi
}

# --- Exp 1: Clean Baseline ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "1" || "$MODE" == "best" ]]; then
    submit_job "sbatch/exp1_baseline_yolo26s.sh" "Exp 1: YOLO26s Baseline"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "1" || "$MODE" == "best" ]]; then
    submit_job "sbatch/exp1_baseline_yolo12s.sh" "Exp 1: YOLO12s Baseline"
fi

# --- Exp 2: Agaian Retinex (Data-Level) ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "2" ]]; then
    submit_job "sbatch/exp2_agaian_retinex_yolo26s.sh" "Exp 2: YOLO26s Agaian Retinex"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "2" ]]; then
    submit_job "sbatch/exp2_agaian_retinex_yolo12s.sh" "Exp 2: YOLO12s Agaian Retinex"
fi

# --- Exp 3: Grand Unified (Data-Level) ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "3" ]]; then
    submit_job "sbatch/exp3_grand_unified_yolo26s.sh" "Exp 3: YOLO26s Grand Unified"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "3" ]]; then
    submit_job "sbatch/exp3_grand_unified_yolo12s.sh" "Exp 3: YOLO12s Grand Unified"
fi

# --- Exp 4: Loss-Only Ablation (Multi-Stage SDice) ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "4" ]]; then
    submit_job "sbatch/exp4_loss_only_yolo26s.sh" "Exp 4: YOLO26s Multi-Stage SDice Loss Only"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "4" ]]; then
    submit_job "sbatch/exp4_loss_only_yolo12s.sh" "Exp 4: YOLO12s Multi-Stage SDice Loss Only"
fi

# --- Exp 5: In-Network 2D Conv Stem ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "5" ]]; then
    submit_job "sbatch/exp5_conv_stem_yolo26s.sh" "Exp 5: YOLO26s In-Network 2D Conv Stem"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "5" ]]; then
    submit_job "sbatch/exp5_conv_stem_yolo12s.sh" "Exp 5: YOLO12s In-Network 2D Conv Stem"
fi

# --- Exp 6: In-Network CTDN Attention Stem ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "6" ]]; then
    submit_job "sbatch/exp6_ctdn_stem_yolo26s.sh" "Exp 6: YOLO26s In-Network CTDN Attention Stem"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "6" ]]; then
    submit_job "sbatch/exp6_ctdn_stem_yolo12s.sh" "Exp 6: YOLO12s In-Network CTDN Attention Stem"
fi

# --- Exp 7: ALL APPROACHES TOGETHER ---
if [[ "$MODE" == "all" || "$MODE" == "yolo26" || "$MODE" == "7" || "$MODE" == "best" ]]; then
    submit_job "sbatch/exp7_all_approaches_yolo26s.sh" "Exp 7: YOLO26s ALL APPROACHES TOGETHER"
fi
if [[ "$MODE" == "all" || "$MODE" == "yolo12" || "$MODE" == "7" || "$MODE" == "best" ]]; then
    submit_job "sbatch/exp7_all_approaches_yolo12s.sh" "Exp 7: YOLO12s ALL APPROACHES TOGETHER"
fi

# --- Exp 8: Master Super-Ensemble ---
if [[ "$MODE" == "8" ]]; then
    submit_job "sbatch/exp8_master_super_ensemble.sh" "Exp 8: Master Super-Ensemble WBF"
fi

echo ""
echo "=========================================================================="
echo "📊 Current Slurm Queue for $USER:"
squeue -u "$USER" || true
echo "=========================================================================="
echo "💡 Tips:"
echo "  • Watch job logs: tail -f slurm_exp*.out"
echo "  • Once models finish training, run the ensemble with:"
echo "      bash sbatch/launch_all_new_experiments.sh 8"
echo "=========================================================================="
