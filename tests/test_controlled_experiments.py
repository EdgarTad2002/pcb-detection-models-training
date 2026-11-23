"""CPU-only tests of experimental controls, leakage checks and training CLI compatibility."""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import yaml
from tools import controlled_experiments as experiments

from tools.controlled_experiments import (
    ROOT, NAMES, audit_datasets, command_for, file_hash, make_jobs,
    matched_budget, prepare, read_dataset, verify_plan,
)


class ControlledExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/controlled_experiments.json").read_text())
        file_hash.cache_clear()

    def test_ablation_pairs_differ_only_in_dataset_and_budget(self):
        jobs = make_jobs(self.config, ["augmentation", "crop"], seeds=[0],
                         counts={"clean": 548, "cappaste": 919, "mixed": 1096})
        self.assertEqual(len(jobs), 3)
        clean = next(j for j in jobs if j["dataset"] == "clean")
        self.assertEqual(clean["studies"], ["augmentation", "crop"])
        self.assertEqual({j["model"] for j in jobs}, {"yolo26s_agwd_sdice"})
        self.assertEqual({self.config["models"][j["model"]]["weights"] for j in jobs}, {"yolo26s.pt"})
        for job in jobs:
            self.assertLessEqual(abs(job["budget"]["relative_difference"]), 0.02)
        self.assertEqual(next(j for j in jobs if j["dataset"] == "mixed")["epochs"], 50)

    def test_main_models_have_same_data_epochs_and_seeds(self):
        jobs = make_jobs(self.config, ["main"])
        self.assertEqual(len(jobs), len(self.config["models"]) * 3)
        self.assertEqual({j["dataset"] for j in jobs}, {"cappaste"})
        self.assertEqual({j["epochs"] for j in jobs}, {100})
        self.assertEqual({j["seed"] for j in jobs}, {0, 1, 2})
        ablation = make_jobs(self.config, ["augmentation"], seeds=[0])
        self.assertFalse({j["key"] for j in jobs} & {j["key"] for j in ablation})

    def test_budget_accounts_for_partial_batches(self):
        budget = matched_budget(548, 919, 8, 100)
        self.assertEqual(budget["target_batches"], 6900)
        self.assertEqual(budget["planned_batches"], 6900)
        self.assertEqual(budget["epochs"], 60)

    def test_submission_defaults_to_independent_jobs_and_optional_limit_adds_dependency(self):
        for limit in (None, 1):
            with self.subTest(limit=limit), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                plan = {"root": tmp, "output": tmp, "jobs": [{"key": "one"}, {"key": "two"}]}
                path = root / "plan.json"
                path.write_text(json.dumps(plan))
                argv = ["controlled_experiments.py", "submit", "--plan", str(path)]
                if limit is not None:
                    argv += ["--max-parallel", str(limit)]
                with patch.object(sys, "argv", argv), patch.object(experiments, "verify_plan"), \
                     patch.object(experiments.subprocess, "run", side_effect=[SimpleNamespace(stdout="101\n"), SimpleNamespace(stdout="102\n")]) as run:
                    experiments.main()
                second = run.call_args_list[1].args[0]
                self.assertEqual("--dependency=afterany:101" in second, limit == 1)
                self.assertEqual(json.loads((root / "submission.json").read_text()), {"one": "101", "two": "102"})

    def test_cancel_targets_only_recorded_plan_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "plan.json"
            path.write_text(json.dumps({"root": tmp, "output": tmp, "jobs": [{"key": "one"}]}))
            record = root / "submission.json"
            record.write_text(json.dumps({"one": "101"}))
            with patch.object(sys, "argv", ["controlled_experiments.py", "cancel", "--plan", str(path)]), \
                 patch.object(experiments.subprocess, "run") as run:
                experiments.main()
                self.assertEqual(run.call_args.args[0], ["scancel", "101"])
                record.write_text(json.dumps({"unrelated": "999"}))
                run.reset_mock()
                with self.assertRaisesRegex(ValueError, "unrelated"):
                    experiments.main()
                run.assert_not_called()

    def test_restart_tag_changes_run_namespace(self):
        tagged = copy.deepcopy(self.config)
        tagged["version"] += "_restart1"
        before = make_jobs(self.config, ["main"], seeds=[0])
        after = make_jobs(tagged, ["main"], seeds=[0])
        self.assertFalse({j["key"] for j in before} & {j["key"] for j in after})

    def dataset(self, root, name, changed_label=False, leak=False):
        folder = root / name
        for split in ("train", "val", "test"):
            images, labels = folder / split / "images", folder / split / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            filename = f"board_{split}.rf.hash"
            (images / f"{filename}.jpg").write_bytes(("val" if leak and split == "train" else split).encode())
            annotation = "0 0.5 0.5 0.1 0.1\n"
            if changed_label and split == "val":
                annotation = "0 0.5 0.5 0.2 0.1\n"
            (labels / f"{filename}.txt").write_text(annotation)
        path = folder / "data.yaml"
        path.write_text(yaml.safe_dump({"path": str(folder), "names": NAMES,
                                       "train": "train/images", "val": "val/images", "test": "test/images"}))
        return path

    def test_accepts_identical_heldout_and_rejects_annotation_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = read_dataset(self.dataset(root, "clean"))
            b = read_dataset(self.dataset(root, "cap"))
            audit_datasets({"clean": a, "cap": b}, "clean")
            c = read_dataset(self.dataset(root, "changed", changed_label=True))
            with self.assertRaisesRegex(ValueError, "val images/labels differ"):
                audit_datasets({"clean": a, "changed": c}, "clean")

    def test_rejects_cross_split_duplicate_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = read_dataset(self.dataset(Path(tmp), "leaky", leak=True))
            with self.assertRaisesRegex(ValueError, "content overlaps"):
                audit_datasets({"clean": ds}, "clean")

    def test_rejects_board_leakage_even_when_pixels_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = read_dataset(self.dataset(Path(tmp), "leaky"))
            ds["splits"]["train"][0]["name"] = "board_val.rf.different_augmentation.jpg"
            with self.assertRaisesRegex(ValueError, "board identifiers overlap"):
                audit_datasets({"clean": ds}, "clean")

    def test_prepare_freezes_lists_and_verify_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = copy.deepcopy(self.config)
            cfg["datasets"]["clean"] = str(self.dataset(root, "clean"))
            cfg["datasets"]["cappaste"] = str(self.dataset(root, "cap"))
            args = argparse.Namespace(study="augmentation", models=None, seeds=[0], output="controlled", device="cpu")
            plan_path = prepare(args, cfg, root)
            plan = json.loads(plan_path.read_text())
            verify_plan(plan)
            snapshot = Path(plan["output"]) / "datasets/clean/val.txt"
            snapshot.write_text(snapshot.read_text().replace("board_val", "missing"))
            with self.assertRaises(FileNotFoundError):
                verify_plan(plan)

    def test_all_generated_commands_parse_and_use_fixed_evaluation(self):
        # Extract CLI functions without importing GPU dependencies unavailable on developer machines.
        tree = ast.parse((ROOT / "train.py").read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"parse_args", "build_train_kwargs"}]
        scope = {"argparse": argparse, "Path": Path, "MODEL_DISPATCH_REGISTRY": {},
                 "DEFAULT_PROJECT_ROOT": ROOT, "DEFAULT_RESULTS_DIR": ROOT / "results"}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "train.py", "exec"), scope)
        jobs = make_jobs(self.config, ["main", "augmentation", "crop"], seeds=[0],
                         counts={"clean": 548, "cappaste": 919, "mixed": 1096})
        for job in jobs:
            cmd = command_for(self.config, job, ROOT, Path("/tmp/protocol"), Path("/tmp/train.yaml"),
                              Path("/tmp/fixed-eval.yaml"))
            with patch.object(sys, "argv", cmd[1:]):
                args = scope["parse_args"]()
            self.assertEqual(args.eval_split, "val")
            self.assertEqual(args.eval_data, Path("/tmp/fixed-eval.yaml"))
            kwargs = scope["build_train_kwargs"](args, args.data)
            self.assertEqual(kwargs["data"], "/tmp/train.yaml")
            self.assertEqual(kwargs["nbs"], kwargs["batch"])
            self.assertEqual(kwargs["copy_paste"], 0)
            self.assertEqual(kwargs["patience"], 0)
            self.assertEqual(kwargs["optimizer"], "SGD")


if __name__ == "__main__":
    unittest.main()
