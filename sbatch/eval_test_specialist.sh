#!/bin/bash
#SBATCH --job-name=eval_spec_test
#SBATCH --partition=research
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_eval_spec_test_%j.out

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

SPECIALIST_WEIGHTS="runs/yolo26s_crop_specialist_mixed1280/pcb-filtered/weights/best.pt"
BASELINE_WEIGHTS="runs/yolo26s_agwd_sdice_smooth0.01_uniform/pcb-filtered/weights/best.pt"
DATA_YAML="datasets/pcb-native-res-cleaned-1280/data.yaml"
TEST_IMAGES="datasets/pcb-native-res-cleaned-1280/test/images"
TEST_LABELS="datasets/pcb-native-res-cleaned-1280/test/labels"

echo "================================================================================"
echo "EVALUATING YOLO26s MIXED CROP SPECIALIST ON TEST SET (44 BOARDS)"
echo "  Weights: $SPECIALIST_WEIGHTS"
echo "================================================================================"

echo ""
echo "--- 1. Standard Ultralytics Test Split Evaluation ---"
python -c '
import sys
sys.path.insert(0, ".")
for mod in ["luma_modules","retinex_stem","weber_stem","squared_dice_loss","agwd_ota_loss","physics_spectral_yolo26"]:
    try: __import__(mod)
    except: pass
from ultralytics import YOLO
model = YOLO("'$SPECIALIST_WEIGHTS'")
res = model.val(data="'$DATA_YAML'", split="test", imgsz=1280, device=0, max_det=1000, verbose=False)
print("\n>>> ULTRALYTICS TEST SPLIT RESULTS <<<")
print(f"Overall mAP@50:    {res.box.map50 * 100:.2f}%")
print(f"Overall mAP50-95: {res.box.map * 100:.2f}%")
for i, cname in enumerate(["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]):
    print(f"  {cname:<24}: AP@50 = {res.box.ap50[i]*100:5.2f}%")
'

echo ""
echo "--- 2. Continuous-Integral AP Evaluation (Apple-to-Apple Baseline Evaluator) ---"
python tools/eval_sahi_native_test.py \
    --weights "$SPECIALIST_WEIGHTS" \
    --test-images "$TEST_IMAGES" \
    --test-labels "$TEST_LABELS" \
    --device 0 \
    --no-slices \
    --max-det 1000

echo ""
echo "--- 3. Dual-Branch Pipeline Evaluation (Full-Board Baseline + Specialist WBF) ---"
python tools/eval_dual_branch_pipeline.py \
    --global-weights "$BASELINE_WEIGHTS" \
    --crop-weights "$SPECIALIST_WEIGHTS" \
    --images-dir "$TEST_IMAGES" \
    --labels-dir "$TEST_LABELS" \
    --split-name "test" \
    --fusion-method "wbf" \
    --merge-iou 0.50 \
    --conf-thresh 0.001 \
    --output-json "runs/dual_branch_eval_test/test_specialist_wbf.json" \
    --device 0

echo "All test evaluations completed!"
