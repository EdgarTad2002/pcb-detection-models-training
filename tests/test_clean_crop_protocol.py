"""End-to-end dataset preparation and paired selection for the no-extra-augmentation study."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import yaml

from tools import build_mixed_crop_dataset as builder
from tools.controlled_experiments import (
    ROOT, NAMES, audit_datasets, command_for, make_jobs, read_dataset,
    select_crop_pair, split_signature,
)


class CleanCropProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/clean_crop_experiments.json").read_text())

    def test_both_models_have_paired_controls_and_matched_budget(self):
        jobs = make_jobs(self.config, ["crop"], counts={"clean": 548, "mixed": 1096})
        self.assertEqual(len(jobs), 12)
        for model in self.config["models"]:
            for seed in (0, 1, 2):
                pair = select_crop_pair({"jobs": jobs}, seed, [model])
                self.assertEqual({j["model"] for j in pair.values()}, {model})
                self.assertEqual(pair["clean"]["epochs"], 100)
                self.assertEqual(pair["mixed"]["epochs"], 50)
                self.assertLess(abs(pair["mixed"]["budget"]["relative_difference"]), .01)
        with self.assertRaisesRegex(ValueError, "Multiple crop models"):
            select_crop_pair({"jobs": jobs}, 0)

    def test_no_additional_photometric_geometric_or_copy_paste_augmentation(self):
        for key in ("copy_paste", "mixup", "mosaic", "degrees", "translate", "scale",
                    "shear", "perspective", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud"):
            self.assertEqual(self.config["common"][key], 0, key)
        jobs = make_jobs(self.config, ["crop"], seeds=[0], counts={"clean": 548, "mixed": 1096})
        for job in jobs:
            command = command_for(self.config, job, ROOT, Path("/tmp/protocol"),
                                  Path(f"/tmp/{job['dataset']}.yaml"), Path("/tmp/clean.yaml"))
            self.assertEqual(command[command.index("--eval-data") + 1], "/tmp/clean.yaml")
            self.assertEqual(command[command.index("--eval-split") + 1], "val")
            self.assertNotIn("--pretrained-weights", command)
            self.assertEqual(command[command.index("--weights") + 1], self.config["models"][job["model"]]["weights"])

    def test_builder_uses_training_only_and_preserves_clean_heldout(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "clean", Path(tmp) / "mixed"
            for index, split in enumerate(("train", "valid", "test")):
                (source / split / "images").mkdir(parents=True)
                (source / split / "labels").mkdir(parents=True)
                for i in range(2):
                    name = f"board_{split}_{i}.rf.hash"
                    pixels = np.random.default_rng(index * 10 + i).integers(0, 255, (128, 128, 3), dtype=np.uint8)
                    Image.fromarray(pixels).save(source / split / "images" / f"{name}.png")
                    (source / split / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
            (source / "data.yaml").write_text(yaml.safe_dump({"path": str(source), "nc": 4, "names": NAMES,
                                                            "train": "train/images", "val": "valid/images", "test": "test/images"}))
            args = argparse.Namespace(source_dataset=str(source), output_dataset=str(output), crop_sizes=[64],
                                      sampling_ratio=.5, min_visibility=.3, min_box_dim=3, seed=42)
            with patch.object(builder, "parse_args", return_value=args), redirect_stdout(io.StringIO()):
                builder.main()
            clean, mixed = read_dataset(source / "data.yaml"), read_dataset(output / "data.yaml")
            audit_datasets({"clean": clean, "mixed": mixed}, "clean")
            self.assertEqual(len(clean["splits"]["train"]), 2)
            self.assertEqual(len(mixed["splits"]["train"]), 4)
            for split in ("val", "test"):
                self.assertEqual(split_signature(clean["splits"][split]), split_signature(mixed["splits"][split]))
            manifest = json.loads((output / "dataset_manifest.json").read_text())
            self.assertTrue(all("board_train_" in sample["filename"] for sample in manifest["train_samples"]))
            # Re-running must not erase the completed dataset.
            with patch.object(builder, "parse_args", return_value=args), redirect_stdout(io.StringIO()):
                with self.assertRaises(FileExistsError):
                    builder.main()
            self.assertTrue((output / "dataset_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
