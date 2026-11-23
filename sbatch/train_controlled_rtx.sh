#!/bin/bash
#SBATCH --job-name=pcb_info17_1280_rtx
#SBATCH --partition=rtx
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/info17_1280_%j.out
#SBATCH --error=logs/info17_1280_%j.err
set -euo pipefail

PLAN="$(realpath "$1")"
JOB_KEY="$2"
export CONDA_PKGS_DIRS=/mnt/weka/etadevosyan/.conda/pkgs
export CONDA_ENVS_PATH=/mnt/weka/etadevosyan/.conda/envs
export YOLO_CONFIG_DIR=/mnt/weka/etadevosyan/.config/Ultralytics
mkdir -p "$YOLO_CONFIG_DIR"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/weka/etadevosyan/.conda/envs/pcb-yolo
PROJECT_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "$PLAN")"
cd "$PROJECT_ROOT"
python -u - "$PLAN" "$JOB_KEY" <<'PYEOF'
import json, shlex, subprocess, sys
from pathlib import Path
plan_path, key = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(Path.cwd()))
from tools.controlled_experiments import verify_plan
plan = json.loads(plan_path.read_text())
verify_plan(plan)
job = next(j for j in plan["jobs"] if j["key"] == key)
argv = [sys.executable, str(Path(plan["root"]) / "train_optionb.py")] + list(job["command"])[2:]
argv[argv.index("--device") + 1] = "0"
print("Running:", shlex.join(argv), flush=True)
subprocess.run(argv, cwd=plan["root"], check=True)
PYEOF
