"""CPU tests for source auditing, equation implementation and training-only pilot references."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
import yaml

from dcd_fciou_loss import paper_fciou_loss
from tools.audit_dcd_source import PAPER_COMMIT, audit_source
from tools.audit_external_pcb_data import audit
from tools.prepare_diffusion_pilot import prepare
from tools.controlled_experiments import NAMES, file_hash


class NextPreparationTest(unittest.TestCase):
    def setUp(self):
        file_hash.cache_clear()

    def dataset(self, root):
        for i, split in enumerate(("train", "val", "test")):
            images, labels = root / split / "images", root / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            pixels = np.random.default_rng(42 + i).integers(0, 255, (128, 128, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(images / f"board_{split}.png", compress_level=0)
            (labels / f"board_{split}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        path = root / "data.yaml"
        path.write_text(yaml.safe_dump({"path": str(root), "names": NAMES, "nc": 4,
                                       "train": "train/images", "val": "val/images", "test": "test/images"}))
        return path

    def test_paper_equation_sign_clamping_and_gradients(self):
        iou = torch.tensor([.02, .2, .5, .8, .98], requires_grad=True)
        ciou = iou - .1
        result = paper_fciou_loss(ciou, iou)
        expected = 1 - ciou + iou - ((iou - .05) / .9).clamp(0, 1)
        self.assertTrue(torch.allclose(result, expected))
        self.assertAlmostEqual(result[1].item(), .93333333, places=6)
        # With the repository helper's opposite score correction, this value would be .8667.
        result.sum().backward()
        self.assertTrue(torch.isfinite(iou.grad).all())
        with self.assertRaises(ValueError):
            paper_fciou_loss(ciou, iou, .95, .05)

    def test_external_audit_detects_reencoded_heldout_and_rotated_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.dataset(root / "reference")
            candidates = root / "external"
            candidates.mkdir()
            with Image.open(root / "reference/test/images/board_test.png") as im:
                im.save(candidates / "different_filename.png", compress_level=9)
            with Image.open(root / "reference/train/images/board_train.png") as im:
                im.rotate(90, expand=True).save(candidates / "rotated.png")
            result = audit(candidates, data, threshold=0)
            self.assertEqual(result["counts"]["candidates_with_exact_heldout_overlap"], 1)
            reencoded = next(f for f in result["findings"] if Path(f["path"]).name == "different_filename.png")
            heldout = next(m for m in reencoded["matches"] if m["split"] == "test")
            self.assertIn("identical_decoded_pixels", heldout["reasons"])
            self.assertNotIn("identical_file", heldout["reasons"])
            rotated = next(f for f in result["findings"] if Path(f["path"]).name == "rotated.png")
            self.assertTrue(any(m["split"] == "train" and m["dhash_distance"] == 0 for m in rotated["matches"]))

    def test_diffusion_preparation_never_uses_heldout_or_claims_generated_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = self.dataset(root / "clean")
            out = root / "pilot"
            result = prepare(data, out, count=2, crop_size=64, seed=42, padding=8)
            self.assertFalse(result["contains_generated_images"])
            self.assertEqual(result["generation_status"], "not_run")
            for sample in result["samples"]:
                self.assertEqual(sample["source_split"], "train")
                self.assertIn("/train/images/", sample["source_image"])
                with Image.open(out / sample["edit_mask"]) as mask:
                    self.assertEqual(mask.getpixel((32, 32)), 0)
                    self.assertEqual(mask.getpixel((0, 0)), 255)
                    self.assertEqual(set(mask.getdata()), {0, 255})
            before = (out / "pilot_manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(data, out, count=2, crop_size=64)
            self.assertEqual(before, (out / "pilot_manifest.json").read_bytes())

    def test_source_audit_detects_ciou_without_executing_external_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contents = {
                "train.py": "raise RuntimeError('must not execute')\n",
                "ultralytics/__init__.py": "__version__ = '8.3.85'\n",
                "ultralytics/cfg/models/DCD.yaml": "nc: 38\n",
                "ultralytics/cfg/models/yolo11.yaml": "nc: 80\n",
                "ultralytics/nn/modules/block.py": "class GSConv: pass\n",
                "ultralytics/tricks/dscconv.py": "class DySnakeConv: pass\n",
                "ultralytics/utils/metrics.py": "def bbox_iou(*args, **kwargs): pass\n",
                "ultralytics/utils/loss.py": "class BboxLoss:\n def forward(self):\n  return bbox_iou(a, b, CIoU=True)\n",
            }
            for name, content in contents.items():
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
            with patch("tools.audit_dcd_source.subprocess.run", side_effect=[
                SimpleNamespace(stdout=PAPER_COMMIT), SimpleNamespace(stdout="ultralytics/__pycache__/x.pyc\n")]):
                result = audit_source(root)
            self.assertTrue(result["author_code_uses_ciou"])
            self.assertFalse(result["author_code_activates_fciou"])
            self.assertFalse(result["tracked_changes"])


if __name__ == "__main__":
    unittest.main()
