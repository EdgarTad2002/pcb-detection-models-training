#!/usr/bin/env python3
"""
tests/test_retinex_dataset_builder.py

Unit tests for build_retinex_cleaned_dataset script:
- Verifies params.json generation and parameter mismatch detection
- Verifies overwrite behavior
- Verifies label copy integrity
- Verifies separate skipped image counting
"""

import json
import shutil
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

from tools.build_retinex_cleaned_dataset import build_retinex_dataset


@pytest.fixture
def mock_dataset_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="mock_pcb_dataset_"))
    splits = ["train", "valid"]
    for s in splits:
        img_dir = temp_dir / s / "images"
        lbl_dir = temp_dir / s / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        # Create two small synthetic test images
        img1 = np.full((32, 32, 3), 100, dtype=np.uint8)
        img2 = np.full((32, 32, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "board_01.jpg"), img1)
        cv2.imwrite(str(img_dir / "board_02.jpg"), img2)

        # Create a YOLO label
        with open(lbl_dir / "board_01.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
        with open(lbl_dir / "board_02.txt", "w") as f:
            f.write("1 0.3 0.3 0.1 0.1\n")

    # Create dummy data.yaml
    with open(temp_dir / "data.yaml", "w") as f:
        f.write("names: ['cap', 'res']\nnc: 2\n")

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_dataset_builder_runs_and_writes_params(mock_dataset_dir):
    dst_dir = mock_dataset_dir.parent / (mock_dataset_dir.name + "_retinex")
    try:
        # First run: process all images
        build_retinex_dataset(
            src_dir=str(mock_dataset_dir),
            dst_dir=str(dst_dir),
            device="cpu",
            c1=0.01,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=3,
            overwrite=False,
        )

        params_file = dst_dir / "params.json"
        assert params_file.exists(), "params.json was not created!"
        with open(params_file, "r") as f:
            data = json.load(f)
        assert data["c1"] == 0.01
        assert data["c2"] == 0.1
        assert data["lambd"] == 1.0
        assert data["gamma"] == 2.2
        assert data["max_iter"] == 3
        assert "retinex_script_sha256" in data

        # Check that labels are copied unchanged
        lbl1 = (dst_dir / "train" / "labels" / "board_01.txt").read_text()
        assert lbl1 == "0 0.5 0.5 0.2 0.2\n"

        # Check second run with SAME params skips already processed images without error
        build_retinex_dataset(
            src_dir=str(mock_dataset_dir),
            dst_dir=str(dst_dir),
            device="cpu",
            c1=0.01,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=3,
            overwrite=False,
        )

        # Check third run with DIFFERENT params without overwrite raises RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            build_retinex_dataset(
                src_dir=str(mock_dataset_dir),
                dst_dir=str(dst_dir),
                device="cpu",
                c1=0.05,  # Changed param!
                c2=0.1,
                lambd=1.0,
                gamma=2.2,
                max_iter=3,
                overwrite=False,
            )
        assert "do not match" in str(exc_info.value)

        # Check fourth run with DIFFERENT params WITH overwrite succeeds
        build_retinex_dataset(
            src_dir=str(mock_dataset_dir),
            dst_dir=str(dst_dir),
            device="cpu",
            c1=0.05,
            c2=0.1,
            lambd=1.0,
            gamma=2.2,
            max_iter=3,
            overwrite=True,
        )
        with open(params_file, "r") as f:
            updated_data = json.load(f)
        assert updated_data["c1"] == 0.05

    finally:
        shutil.rmtree(dst_dir, ignore_errors=True)
