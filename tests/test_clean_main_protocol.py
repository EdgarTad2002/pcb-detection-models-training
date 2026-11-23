"""Verify clean-only coverage and safe reuse of the two existing controls."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml
from tools.controlled_experiments import ROOT, NAMES, file_hash, make_jobs, read_dataset, split_signature
from tools import prepare_clean_main as clean_main


class CleanMainProtocolTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/clean_main_experiments.json").read_text())
        self.crop = json.loads((ROOT / "configs/clean_crop_experiments.json").read_text())
        file_hash.cache_clear()

    def test_clean_only_seven_models_three_seeds_and_matching_controls(self):
        main = json.loads((ROOT / "configs/controlled_experiments.json").read_text())
        jobs = make_jobs(self.config, ["main"])
        self.assertEqual(len(jobs), 21)
        self.assertEqual({j["dataset"] for j in jobs}, {"clean"})
        self.assertEqual({j["epochs"] for j in jobs}, {100})
        self.assertEqual({j["seed"] for j in jobs}, {0, 1, 2})
        self.assertEqual(self.config["common"], self.crop["common"])
        self.assertEqual(set(self.config["datasets"]), {"clean"})
        for model in self.config["models"]:
            self.assertEqual(self.config["models"][model], main["models"][model])
        for key in ("mosaic", "mixup", "copy_paste", "degrees", "translate", "scale", "shear",
                    "perspective", "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud"):
            self.assertEqual(self.config["common"][key], 0, key)
        self.assertEqual(self.config["evaluation_dataset"], "clean")

    def fixture(self, root):
        for split in ("train", "val", "test"):
            folder = root / "datasets/pcb-native-res-cleaned-1280" / split
            (folder / "images").mkdir(parents=True)
            (folder / "labels").mkdir(parents=True)
            (folder / "images" / f"board_{split}.jpg").write_bytes(split.encode())
            (folder / "labels" / f"board_{split}.txt").write_text("0 .5 .5 .2 .2\n")
        data = root / self.config["datasets"]["clean"]
        data.write_text(yaml.safe_dump({"path": str(data.parent), "names": NAMES,
                                        **{s: s + "/images" for s in ("train", "val", "test")}}))
        dataset = read_dataset(data)
        jobs = make_jobs(self.crop, ["crop"], seeds=[0], counts={"clean": 548, "mixed": 1096})
        return {"root": str(root), "output": str(root / "old_control"), "config": self.crop,
                "dataset_fingerprints": {"clean": {s: split_signature(rs) for s, rs in dataset["splits"].items()}},
                "jobs": jobs}

    def test_bundle_prepares_19_jobs_and_keeps_original_control_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = self.fixture(root)
            (root / "configs").mkdir()
            (root / "configs/clean_main_experiments.json").write_text(json.dumps(self.config))
            source = root / "control.json"
            source.write_text(json.dumps(control))
            with patch.object(sys, "argv", ["prepare_clean_main.py", "--project-root", str(root),
                                             "--control-plan", str(source)]), \
                    patch.object(clean_main.subprocess, "run") as run, redirect_stdout(io.StringIO()):
                clean_main.main()
            run.assert_not_called()
            bundle = json.loads((root / "controlled_experiments/clean_main_noextraaug_v1_bundle.json").read_text())
            plans = [json.loads(Path(p).read_text()) for p in bundle["new_plans"]]
            self.assertEqual([len(p["jobs"]) for p in plans], [15, 4])
            new = {(j["model"], j["seed"]) for p in plans for j in p["jobs"]}
            reused = {(j["model"], j["seed"]) for j in bundle["reused_controls"]}
            self.assertFalse(new & reused)
            self.assertEqual(new | reused, {(m, s) for m in self.config["models"] for s in (0, 1, 2)})
            self.assertTrue(all("old_control/validation" in j["validation_result"] for j in bundle["reused_controls"]))

    def test_reuse_rejects_recipe_or_dataset_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = self.fixture(root)
            changed = copy.deepcopy(plan)
            changed["config"]["common"]["hsv_v"] = .4
            with self.assertRaisesRegex(ValueError, "common"):
                clean_main.validate_controls(self.config, changed, root)
            plan["dataset_fingerprints"]["clean"]["train"] = "different"
            with self.assertRaisesRegex(ValueError, "dataset changed"):
                clean_main.validate_controls(self.config, plan, root)

    def test_submission_preflights_both_plans_before_any_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            control = self.fixture(root)
            (root / "configs").mkdir()
            (root / "configs/clean_main_experiments.json").write_text(json.dumps(self.config))
            source = root / "control.json"
            source.write_text(json.dumps(control))
            argv = ["prepare_clean_main.py", "--project-root", str(root),
                    "--control-plan", str(source), "--submit"]
            with patch.object(sys, "argv", argv), patch.object(clean_main.subprocess, "run") as run, \
                    redirect_stdout(io.StringIO()):
                clean_main.main()
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    self.assertIn("submit", call.args[0])
                    self.assertNotIn("--max-parallel", call.args[0])
                bundle = json.loads((root / "controlled_experiments/clean_main_noextraaug_v1_bundle.json").read_text())
                (Path(bundle["new_plans"][1]).parent / "submission.json").write_text('{}')
                run.reset_mock()
                with self.assertRaisesRegex(FileExistsError, "already submitted"):
                    clean_main.main()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
