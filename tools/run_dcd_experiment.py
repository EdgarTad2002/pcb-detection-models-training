#!/usr/bin/env python3
"""Isolated DCD-Net smoke/training entrypoint. Never installs or replaces packages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.audit_dcd_source import PAPER_COMMIT, audit_source


def run(args):
    import yaml
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    audit = audit_source(args.source)
    if audit["commit"] != PAPER_COMMIT or audit["tracked_changes"]:
        raise ValueError("Use the inspected immutable author commit with no tracked modifications")
    if audit["ultralytics_version"] != "8.3.85":
        raise ValueError("Unexpected author package version")
    args.output.mkdir(parents=True)
    (args.output / "source_audit.json").write_text(json.dumps(audit, indent=2))
    # The author source is scoped to this process; installed pcb-yolo is not modified.
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    torch.set_num_threads(args.cpu_threads)
    import ultralytics
    from ultralytics import YOLO
    if not Path(ultralytics.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError("Wrong Ultralytics imported; refusing mixed source/package execution")
    if args.mode == "dcd_paper_fc":
        from dcd_fciou_loss import enable_paper_fciou
        enable_paper_fciou(args.fc_lower, args.fc_upper)
    config_path = args.source / "ultralytics/cfg/models" / ("yolo11.yaml" if args.mode == "yolo11n" else "DCD.yaml")
    model_cfg = yaml.safe_load(config_path.read_text())
    model_cfg.update(nc=4, scale="n")
    cfg_output = args.output / "model.yaml"
    cfg_output.write_text(yaml.safe_dump(model_cfg, sort_keys=False))
    torch.manual_seed(args.seed)
    model = YOLO(str(cfg_output), task="detect")
    metadata = {"mode": args.mode, "action": args.action, "seed": args.seed,
                "source_commit": audit["commit"], "ultralytics_version": ultralytics.__version__,
                "ultralytics_source": str(ultralytics.__file__), "torch_version": str(torch.__version__),
                "initialization": "random; pretrained=False; no PCB checkpoints",
                "loss": "paper equation 5 adaptation" if args.mode == "dcd_paper_fc" else "author code CIoU",
                "fc_lower": args.fc_lower if args.mode == "dcd_paper_fc" else None,
                "fc_upper": args.fc_upper if args.mode == "dcd_paper_fc" else None,
                "parameters": sum(p.numel() for p in model.model.parameters())}
    (args.output / "provenance.json").write_text(json.dumps(metadata, indent=2))
    if args.action == "smoke":
        from ultralytics.cfg import get_cfg
        # The normal trainer supplies a namespace; YAML-only construction leaves a dict.
        model.model.args = get_cfg(model.model.args)
        image = torch.rand(2, 3, 128, 128)
        model.model.train()
        batch = {"img": image, "batch_idx": torch.tensor([0, 1]),
                 "cls": torch.tensor([[0.], [1.]]),
                 "bboxes": torch.tensor([[.5, .5, .2, .2], [.4, .4, .2, .2]])}
        loss, items = model.model.loss(batch)
        loss = loss.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke loss")
        loss.backward()
        gradients = [p.grad for p in model.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("Missing gradients or non-finite gradients")
        with tempfile.TemporaryDirectory(prefix="pcb-dcd-checkpoint-") as tmp:
            checkpoint = Path(tmp) / "smoke.pt"
            model.save(str(checkpoint))
            reloaded = YOLO(str(checkpoint), task="detect")
            reloaded.model.eval()
            with torch.no_grad():
                reloaded.model(image[:1])
        summary = {**metadata, "status": "smoke_passed", "loss_value": float(loss.detach()),
                   "backward_checked": True, "checkpoint_reload_checked": True}
    else:
        if args.data is None or args.eval_data is None:
            raise ValueError("Training requires --data and --eval-data")
        from tools.controlled_experiments import read_dataset, audit_datasets, split_signature
        training = read_dataset(args.data)
        evaluation = read_dataset(args.eval_data)
        audit_datasets({"train_source": training, "canonical": evaluation}, "canonical")
        # Start from the explicit no-extra-augmentation recipe used by the clean/crop pilot.
        protocol = json.loads((ROOT / "configs/clean_crop_experiments.json").read_text())["common"]
        options = {k: v for k, v in protocol.items() if k not in {"no_cache", "eval_conf", "eval_iou"}}
        options.update(cache=False, conf=protocol["eval_conf"], iou=protocol["eval_iou"],
                       pretrained=False, data=str(args.data.resolve()), epochs=args.epochs,
                       imgsz=args.imgsz, plots=not args.no_plots,
                       device=args.device, seed=args.seed, val=True, split="val",
                       project=str(args.output / "training"), name="fit", exist_ok=False)
        # Freeze validation/test membership to the same canonical lists in a private YAML.
        ds_dir = args.output / "dataset"
        ds_dir.mkdir()
        for split in ("train", "val", "test"):
            records = training["splits"][split] if split == "train" else evaluation["splits"][split]
            (ds_dir / f"{split}.txt").write_text("".join(r["image"] + "\n" for r in records))
        ds_cfg = {"path": str(ds_dir.resolve()), "nc": 4, "names": ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"],
                  **{s: str((ds_dir / f"{s}.txt").resolve()) for s in ("train", "val", "test")}}
        ds_yaml = ds_dir / "data.yaml"
        ds_yaml.write_text(yaml.safe_dump(ds_cfg))
        options["data"] = str(ds_yaml.resolve())
        (args.output / "training_options.json").write_text(json.dumps(options, indent=2))
        (args.output / "dataset_fingerprints.json").write_text(json.dumps({
            "training": {s: split_signature(rs) for s, rs in training["splits"].items()},
            "evaluation": {s: split_signature(rs) for s, rs in evaluation["splits"].items()}}, indent=2))
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        (args.output / "environment.txt").write_text(freeze.stdout)
        model.train(**options)
        checkpoint = Path(model.trainer.best)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        evaluated = YOLO(str(checkpoint), task="detect")
        metrics = evaluated.val(data=str(ds_yaml.resolve()), split="val", imgsz=args.imgsz,
                                batch=protocol["batch"], conf=protocol["eval_conf"], iou=protocol["eval_iou"],
                                max_det=protocol["max_det"], half=False, device=args.device,
                                project=str(args.output / "evaluation"), name="val", plots=not args.no_plots)
        summary = {**metadata, "status": "training_and_validation_completed", "split": "val",
                   "weights": str(checkpoint.resolve()), "epochs": args.epochs,
                   "imgsz": args.imgsz,
                   "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
                   "per_class_ap50": {str(metrics.names[int(i)]): float(ap) for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap50)},
                   "per_class_ap50_95": {str(metrics.names[int(i)]): float(ap) for i, ap in zip(metrics.box.ap_class_index, metrics.box.ap)},
                   "note": "Scratch-trained under author Ultralytics 8.3.85; do not claim initialization/framework equivalence with pretrained 8.4.126 runs."}
    summary["recorded_unix_time"] = time.time()
    (args.output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["yolo11n", "dcd_code", "dcd_paper_fc"], required=True)
    p.add_argument("--action", choices=["smoke", "train"], default="smoke")
    p.add_argument("--data", type=Path)
    p.add_argument("--eval-data", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--fc-lower", type=float, default=.05)
    p.add_argument("--fc-upper", type=float, default=.95)
    p.add_argument("--cpu-threads", type=int, default=2)
    args = p.parse_args()
    if args.epochs < 1 or args.cpu_threads < 1 or args.imgsz < 64 or args.imgsz % 32:
        p.error("epochs/cpu-threads must be positive; imgsz must be >=64 and divisible by 32")
    run(args)


if __name__ == "__main__":
    main()
