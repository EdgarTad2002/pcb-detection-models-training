#!/bin/bash
# ==============================================================================
# submit_retinex_study.sh - submit the bilateral-Retinex experiment matrix.
#
# Usage (run on the cluster, or via: bash tools/remote_train.sh submit_retinex_study.sh <tier>):
#   bash sbatch/submit_retinex_study.sh core     # 8 jobs  (A noise + B hsv)   <- start here
#   bash sbatch/submit_retinex_study.sh grid     # 4 jobs  (C alpha/beta ablation)
#   bash sbatch/submit_retinex_study.sh extras   # 4 jobs  (D copy-paste, stem, x-model, P2)
#   bash sbatch/submit_retinex_study.sh all      # 16 jobs
#
# Question each tier answers
#   core   : is bilateral better than baseline AT ALL, given seed noise?  does lowering
#            hsv_v (which re-randomizes brightness after Retinex) help?
#   grid   : do the Canny edges (beta) help?  is alpha=0.3 a sensible fusion weight?
#   extras : copy-paste capacitors / learnable edge stem / YOLO26x / P2 small-object head
#
# Every run uses --max-det 1000 and the run key encodes all parameters. The baseline
# controls are re-run with the same settings so comparisons are like-for-like (the old
# cleaned_1280 baseline used max_det=300).
# ==============================================================================
set -e
TIER=${1:-core}
S=sbatch/train_bilateral_sweep.sh   #  MODE ALPHA BETA SEED HSV_V WEIGHTS

core () {
    for SEED in 0 1 2; do
        sbatch $S base      0.30 0.10 $SEED 0.4 yolo26s.pt
        sbatch $S bilateral 0.30 0.10 $SEED 0.4 yolo26s.pt
    done
    sbatch $S base      0.30 0.10 0 0.1 yolo26s.pt
    sbatch $S bilateral 0.30 0.10 0 0.1 yolo26s.pt
}
grid () {
    sbatch $S bilateral 0.30 0.00 0 0.4 yolo26s.pt   # beta=0: do edges matter?
    sbatch $S bilateral 0.30 0.20 0 0.4 yolo26s.pt   # stronger edges
    sbatch $S bilateral 0.15 0.10 0 0.4 yolo26s.pt   # weaker reflectance
    sbatch $S bilateral 0.50 0.10 0 0.4 yolo26s.pt   # stronger reflectance
}
extras () {
    sbatch $S cappaste  0.30 0.10 0 0.4 yolo26s.pt
    sbatch $S stem      0.30 0.10 0 0.4 yolo26s.pt
    sbatch $S bilateral 0.30 0.10 0 0.4 yolo26x.pt
    sbatch $S bilateral 0.30 0.10 0 0.4 yolo26s-p2.yaml
}

case "$TIER" in
    core) core ;;
    grid) grid ;;
    extras) extras ;;
    all) core; grid; extras ;;
    *) echo "unknown tier '$TIER' (core|grid|extras|all)"; exit 2 ;;
esac
echo "Submitted tier '$TIER'. Watch with: squeue -u etadevosyan"
