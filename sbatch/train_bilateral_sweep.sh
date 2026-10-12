#!/bin/bash
#SBATCH --job-name=retinex_sweep
#SBATCH --partition=research
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_sweep_%j.out

# ==============================================================================
# Parametrized experiment runner for the bilateral-Retinex study.
#
# Usage (positional, all optional):
#   sbatch sbatch/train_bilateral_sweep.sh MODE ALPHA BETA SEED HSV_V WEIGHTS [extra train.py args...]
#
#   MODE     base            plain cleaned dataset (control)
#            bilateral       Bilateral Retinex fusion  (1-a)*I + a*R + b*Edge
#            cappaste        bbox capacitor copy-paste on cleaned, THEN bilateral
#            stem            plain cleaned dataset + learnable in-network Retinex-Edge stem
#   ALPHA/BETA  fusion weights (ignored for base/stem)         default 0.30 / 0.10
#   SEED     training seed                                      default 0
#   HSV_V    Ultralytics hsv_v brightness jitter (default 0.4)
#   WEIGHTS  yolo26s.pt | yolo26x.pt | yolo26s-p2.yaml (auto warm-starts from yolo26s.pt)
#
# Every dataset is built once into datasets/<name> (name encodes the params) under a
# file lock, so concurrent jobs sharing a config never race. All runs use
# --max-det 1000 so dense boards are not truncated at 300 detections.
# ==============================================================================

MODE=${1:-bilateral}
ALPHA=${2:-0.30}
BETA=${3:-0.10}
SEED=${4:-0}
HSV_V=${5:-0.4}
WEIGHTS=${6:-yolo26s.pt}
EXTRA="${@:7}"

export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"

source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo

export PYTHONPATH="/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training:$PYTHONPATH"
cd /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training

CLEANED=datasets/pcb-unified-4class-cleaned
ARCH_TAG=$(basename "${WEIGHTS%.*}")
TRAIN_FLAGS=""
if [[ "$WEIGHTS" == *.yaml ]]; then
    TRAIN_FLAGS="$TRAIN_FLAGS --pretrained-weights yolo26s.pt"
fi

build_bilateral () {  # $1=source dataset  $2=dest dataset
    local SRC=$1 DST=$2
    mkdir -p datasets
    (
        flock 9
        if [ ! -f "$DST/data.yaml" ]; then
            echo ">>> building $DST from $SRC (alpha=$ALPHA beta=$BETA)"
            python tools/enhance_pcb_retinex.py --source "$SRC" --dest "$DST" \
                --alpha "$ALPHA" --beta "$BETA" --workers 8
        else
            echo ">>> reusing existing $DST"
        fi
    ) 9>"datasets/.lock-$(basename "$DST")"
}

case "$MODE" in
    base)
        DATA=$CLEANED/data.yaml
        RUN_KEY="sweep_base_s${SEED}_hv${HSV_V}_${ARCH_TAG}"
        ;;
    stem)
        DATA=$CLEANED/data.yaml
        RUN_KEY="sweep_stem_s${SEED}_hv${HSV_V}_${ARCH_TAG}"
        TRAIN_FLAGS="$TRAIN_FLAGS --retinex-stem"
        ;;
    bilateral)
        DST=datasets/pcb-bilat-a${ALPHA}-b${BETA}
        build_bilateral "$CLEANED" "$DST"
        DATA=$DST/data.yaml
        RUN_KEY="sweep_bilat_a${ALPHA}_b${BETA}_s${SEED}_hv${HSV_V}_${ARCH_TAG}"
        ;;
    cappaste)
        PASTED=datasets/pcb-cleaned-cappaste
        (
            flock 9
            if [ ! -f "$PASTED/data.yaml" ]; then
                echo ">>> building capacitor bank + copy-paste dataset"
                python build_capacitor_bank.py --source "$CLEANED" --dest capacitor_bank_cleaned --margin 0.15
                python bbox_copy_paste.py --source "$CLEANED" --bank capacitor_bank_cleaned \
                    --dest "$PASTED" --eval-source "$CLEANED" --paste-prob 0.7
                # bbox_copy_paste only writes train/ and points val/test at the RAW images.
                # Link valid/test in and write a standard data.yaml so the bilateral step
                # processes ALL splits (otherwise eval would run on unprocessed images).
                for S in valid test; do
                    [ -e "$PASTED/$S" ] || ln -s "$(realpath $CLEANED/$S)" "$PASTED/$S"
                done
                python - "$CLEANED" "$PASTED" <<'PY'
import sys, yaml
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load((src / "data.yaml").read_text())
out = {"path": str(dst.resolve()), "train": "train/images", "val": "valid/images",
       "test": "test/images", "nc": cfg["nc"], "names": cfg["names"]}
(dst / "data.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
PY
            fi
        ) 9>"datasets/.lock-pcb-cleaned-cappaste"
        DST=datasets/pcb-cappaste-bilat-a${ALPHA}-b${BETA}
        build_bilateral "$PASTED" "$DST"
        DATA=$DST/data.yaml
        RUN_KEY="sweep_cappaste_bilat_a${ALPHA}_b${BETA}_s${SEED}_hv${HSV_V}_${ARCH_TAG}"
        ;;
    *)
        echo "Unknown MODE: $MODE (use base|bilateral|cappaste|stem)"; exit 2 ;;
esac

# Guard against the data.yaml path bug that silently trained on the wrong dataset before.
DATA_ROOT=$(grep '^path:' "$DATA" | awk '{print $2}')
EXPECT=$(realpath "$(dirname "$DATA")")
if [ "$(realpath "$DATA_ROOT")" != "$EXPECT" ]; then
    echo "FATAL: $DATA has path: $DATA_ROOT but lives in $EXPECT -- would train on the wrong images."
    exit 3
fi

echo "=========================================================================="
echo "RUN_KEY=$RUN_KEY  MODE=$MODE  DATA=$DATA  SEED=$SEED  HSV_V=$HSV_V  WEIGHTS=$WEIGHTS"
echo "=========================================================================="

python train.py \
    --run-key "$RUN_KEY" \
    --weights "$WEIGHTS" \
    --data "$DATA" \
    --epochs 100 --imgsz 1280 --batch 8 --workers 8 \
    --seed "$SEED" --hsv-v "$HSV_V" --max-det 1000 \
    --eval-conf 0.001 $TRAIN_FLAGS $EXTRA
