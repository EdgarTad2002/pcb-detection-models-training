"""Regression tests for tools/enhance_pcb_retinex.py dataset generation."""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _make_ds(root: Path):
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        img = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(root / split / "images" / f"{split}.jpg"), img)
        (root / split / "labels" / f"{split}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text(yaml.safe_dump({
        "path": str(root), "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": 4, "names": {0: "a", 1: "b", 2: "c", 3: "d"}}))


def test_data_yaml_path_points_at_destination_even_if_names_overlap(tmp_path):
    # dest name CONTAINS the source name -- the old chained str.replace produced 'x_bilat_bilat'
    src, dst = tmp_path / "x", tmp_path / "x_bilat"
    _make_ds(src)
    subprocess.run([sys.executable, "tools/enhance_pcb_retinex.py", "--source", str(src),
                    "--dest", str(dst), "--workers", "1"], cwd=ROOT, check=True)
    cfg = yaml.safe_load((dst / "data.yaml").read_text())
    assert Path(cfg["path"]) == dst.resolve()
    assert (cfg["train"], cfg["val"], cfg["test"]) == ("train/images", "valid/images", "test/images")
    # labels copied unchanged, same image count per split
    for split in ("train", "valid", "test"):
        assert (dst / split / "labels" / f"{split}.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
        assert len(list((dst / split / "images").glob("*.jpg"))) == 1
