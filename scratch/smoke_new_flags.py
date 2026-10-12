#!/usr/bin/env python3
"""CPU smoke test for the new train.py flags + bilateral dataset builder (no cluster needed)."""
import json, shutil, subprocess, sys
from pathlib import Path
import cv2, numpy as np, yaml

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "scratch" / "mini_smoke"
shutil.rmtree(WORK, ignore_errors=True)
src = WORK / "ds_raw"
rng = np.random.default_rng(0)
sample = sorted((ROOT / "data_samples" / "native_images").glob("*.jpg"))[:3]
assert sample, "need data_samples/native_images/*.jpg"

for split in ("train", "valid", "test"):
    (src / split / "images").mkdir(parents=True)
    (src / split / "labels").mkdir(parents=True)
    for i, p in enumerate(sample):
        im = cv2.resize(cv2.imread(str(p)), (320, 320))
        cv2.imwrite(str(src / split / "images" / f"{split}{i}.jpg"), im)
        lines = [f"{rng.integers(0,4)} {rng.uniform(.2,.8):.4f} {rng.uniform(.2,.8):.4f} 0.1 0.1" for _ in range(5)]
        (src / split / "labels" / f"{split}{i}.txt").write_text("\n".join(lines))
(src / "data.yaml").write_text(yaml.safe_dump({
    "path": str(src), "train": "train/images", "val": "valid/images", "test": "test/images",
    "nc": 4, "names": {0: "Capacitor", 1: "Connector", 2: "Electrolytic Capacitor", 3: "IC"}}))

dst = WORK / "ds_bilat"
subprocess.run([sys.executable, "tools/enhance_pcb_retinex.py", "--source", str(src), "--dest", str(dst),
                "--alpha", "0.3", "--beta", "0.1", "--workers", "2"], cwd=ROOT, check=True)
cfg = yaml.safe_load((dst / "data.yaml").read_text())
assert Path(cfg["path"]).resolve() == dst.resolve(), f"data.yaml path bug: {cfg['path']}"
print("OK data.yaml path points at destination:", cfg["path"])

def run(key, *extra):
    cmd = [sys.executable, "train.py", "--run-key", key, "--weights", "yolo26s.pt",
           "--data", str(dst / "data.yaml"), "--epochs", "1", "--imgsz", "160", "--batch", "2",
           "--workers", "0", "--device", "cpu", "--no-cache", "--project-root", str(WORK),
           "--results-dir", str(WORK / "results"), "--seed", "7", "--max-det", "1000", "--hsv-v", "0.1", *extra]
    subprocess.run(cmd, cwd=ROOT, check=True)
    out = json.loads((WORK / "results" / f"{key}.json").read_text())
    assert out["seed"] == 7 and out["max_det"] == 1000 and out["hsv_v"] == 0.1, out
    print(f"OK {key}: seed/max_det/hsv_v recorded, retinex_stem={out['retinex_stem']}")

run("smoke_plain")
run("smoke_stem", "--retinex-stem")
print("ALL SMOKE CHECKS PASSED")
