#!/bin/bash
# ==============================================================================
# Submit standard alpha parameter sweep for Physics Spectral YOLO26s on
# native resolution rectified 4-class dataset (1280px).
# Alpha grid: [0.0, 0.25, 0.50, 1.00]
# ==============================================================================

ALPHAS=("0.0" "0.25" "0.50" "1.00")

echo "=================================================================="
echo "Submitting Native-Res Physics Spectral sweep across alphas: ${ALPHAS[*]}"
echo "Dataset: datasets/pcb-native-res-unified-4class/data.yaml (1280px)"
echo "=================================================================="

for ALPHA in "${ALPHAS[@]}"; do
    echo "Submitting job for alpha = $ALPHA..."
    sbatch sbatch/train_native_res_physics_spectral.sh "$ALPHA"
    sleep 2
done

echo ""
echo "All 4 jobs successfully queued!"
