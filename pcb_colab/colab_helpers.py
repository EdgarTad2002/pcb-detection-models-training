"""
Helpers for training PCB YOLO models on Google Colab.

Usage (after cloning the repo in Colab):
    from pcb_colab.colab_helpers import *
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from ultralytics import YOLO

# Original Roboflow class IDs kept during training (4 PCB components).
FILTER_CLASS_IDS = [2, 4, 7, 9]
CLASS_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]

MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "yolov5s": {
        "weights": "yolov5s.pt",
        "project": "runs/yolov5_ultralytics",
        "name": "yolov5s-ultralytics",
    },
    "yolov8s": {
        "weights": "yolov8s.pt",
        "project": "runs/yolov8",
        "name": "pcb-filtered",
    },
    "yolov9s": {
        "weights": "yolov9s.pt",
        "project": "runs/yolov9s_ultralytics",
        "name": "yolov9s-ultralytics",
    },
    "yolov10s": {
        "weights": "yolov10s.pt",
        "project": "runs/yolov10",
        "name": "pcb-filtered",
    },
    "yolov11s": {
        "weights": "yolo11s.pt",
        "project": "runs/yolov11",
        "name": "pcb-filtered",
    },
    "yolov12s": {
        "weights": "yolo12s.pt",
        "project": "runs/yolov12",
        "name": "pcb-filtered",
    },
    "yolov26s": {
        "weights": "yolo26s.pt",
        "project": "runs/yolov26",
        "name": "pcb-filtered",
    },
}


def get_project_root() -> Path:
    """Return repo root whether code runs locally or from /content."""
    here = Path(__file__).resolve().parent
    return here.parent


def dataset_yaml(project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    return root / "datasets" / "pcb-filtered-yolov8" / "data.yaml"


def run_dir_for(model_key: str, project_root: Path | None = None) -> Path:
    root = project_root or get_project_root()
    cfg = MODEL_CONFIGS[model_key]
    return root / cfg["project"] / cfg["name"]


def prepare_dataset(
    roboflow_download_dir: str | Path,
    project_root: Path | None = None,
    dest_folder_name: str = "pcb-filtered-yolov8",
) -> Path:
    """
    Copy Roboflow export into the path expected by training/ensemble scripts.
    """
    root = project_root or get_project_root()
    src = Path(roboflow_download_dir).resolve()
    if not src.exists():
        raise FileNotFoundError(f"Roboflow dataset folder not found: {src}")

    dst = root / "datasets" / dest_folder_name

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # The YOLOv11 symlink workaround only makes sense for the original
    # 640px dataset -- skip it for any alternate destination.
    if dest_folder_name == "pcb-filtered-yolov8":
        yolov11_dst = root / "datasets" / "pcb-filtered-yolov11"
        if yolov11_dst.exists() or yolov11_dst.is_symlink():
            yolov11_dst.unlink()
        yolov11_dst.symlink_to(dst.resolve())
        print(f"YOLOv11 symlink: {yolov11_dst} -> {dst}")

    data_yaml = dst / "data.yaml"
    if data_yaml.exists():
        with data_yaml.open() as f:
            data = yaml.safe_load(f)
        data["path"] = str(dst.resolve())
        with data_yaml.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

    print(f"Dataset ready at: {dst}")
    return dst


def merge_ic_classes(dataset_root: Path) -> int:
    """Merge class 22 ('iC') into class 9 ('IC') across all splits.

    The Roboflow source data has inconsistent capitalization for the IC class
    (index 9 = 'IC', index 22 = 'iC'), splitting what should be one class into
    two. This remaps every 'iC' label to 'IC' in place.

    Returns the number of label lines remapped.
    """
    remapped_lines = 0
    for split in ("train", "valid", "test"):
        label_dir = dataset_root / split / "labels"
        if not label_dir.exists():
            continue
        for fpath in label_dir.glob("*.txt"):
            lines = fpath.read_text().splitlines()
            new_lines = []
            changed = False
            for line in lines:
                parts = line.split()
                if parts and parts[0] == "22":
                    parts[0] = "9"
                    changed = True
                    remapped_lines += 1
                new_lines.append(" ".join(parts))
            if changed:
                fpath.write_text("\n".join(new_lines) + "\n")
    return remapped_lines


def download_roboflow_dataset(
    api_key: str,
    project_root: Path | None = None,
    version: int = 3,
    dest_folder_name: str = "pcb-filtered-yolov8",
) -> Path:
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-100").project("printed-circuit-board")
    dataset = project.version(version).download("yolov8")
    dataset_root = prepare_dataset(
        dataset.location,
        project_root=project_root,
        dest_folder_name=dest_folder_name,
    )

    n = merge_ic_classes(dataset_root)
    print(f"Merged {n} 'iC' labels into 'IC' (class 22 -> 9)")

    return dataset_root


def train_model(
    model_key: str,
    *,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str | int = 0,
    workers: int = 2,
    project_root: Path | None = None,
) -> Any:
    """Train one YOLO variant with the same settings used in the paper."""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from: {list(MODEL_CONFIGS)}")

    root = project_root or get_project_root()
    data_yaml = dataset_yaml(root)
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Missing {data_yaml}. Run download_roboflow_dataset() first."
        )

    cfg = MODEL_CONFIGS[model_key]
    print("=" * 70)
    print(f"Training {model_key} on GPU {device}")
    print(f"Data: {data_yaml}")
    print(f"Weights: {cfg['weights']}")
    print("=" * 70)

    model = YOLO(cfg["weights"])
    results = model.train(
        data=str(data_yaml),
        classes=FILTER_CLASS_IDS,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(root / cfg["project"]),
        name=cfg["name"],
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        verbose=True,
        device=device,
        workers=workers,
        patience=100,
        cache=True,
        plots=True,
        save=True,
        val=True,
    )
    return results


def plot_training_curves(results_csv: str | Path, title: str = "Training curves") -> None:
    csv_path = Path(results_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train/box_loss" in df.columns:
        axes[0].plot(df["epoch"], df["train/box_loss"], label="train box")
        if "val/box_loss" in df.columns:
            axes[0].plot(df["epoch"], df["val/box_loss"], label="val box")
        axes[0].set_title("Box loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()

    metric_cols = [c for c in df.columns if "mAP50" in c]
    if metric_cols:
        for col in metric_cols:
            axes[1].plot(df["epoch"], df[col], label=col)
        axes[1].set_title("Validation mAP")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def validate_model(
    model_key: str,
    weights_path: str | Path | None = None,
    project_root: Path | None = None,
) -> dict[str, float]:
    root = project_root or get_project_root()
    weights = Path(weights_path) if weights_path else run_dir_for(model_key, root) / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    metrics = model.val(data=str(dataset_yaml(root)), device=0)
    summary = {
        "model": model_key,
        "weights": str(weights),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    print(json.dumps(summary, indent=2))
    return summary


def analyze_model_results(
    model_key: str,
    results_csv: str | Path | None = None,
    weights_path: str | Path | None = None,
    data_yaml_path: str | Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Plot curves, run validation, and return a compact metrics dict."""
    root = project_root or get_project_root()
    run_dir = run_dir_for(model_key, root)
    csv_path = Path(results_csv) if results_csv else run_dir / "results.csv"
    weights = Path(weights_path) if weights_path else run_dir / "weights" / "best.pt"
    _ = Path(data_yaml_path) if data_yaml_path else dataset_yaml(root)

    plot_training_curves(csv_path, title=f"{model_key} training")
    metrics = validate_model(model_key, weights_path=weights, project_root=root)
    return metrics


def predict_examples(
    model_key: str,
    image_dir: str | Path,
    *,
    weights_path: str | Path | None = None,
    conf: float = 0.25,
    save_dir: str | Path = "colab_outputs/predictions",
    project_root: Path | None = None,
) -> Path:
    root = project_root or get_project_root()
    weights = Path(weights_path) if weights_path else run_dir_for(model_key, root) / "weights" / "best.pt"
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    results = model.predict(
        source=str(image_dir),
        conf=conf,
        save=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        device=0,
    )

    print(f"Saved predictions under: {out_dir}")
    print(f"Processed {len(results)} image(s)")
    return out_dir


def backup_run_to_drive(
    model_key: str,
    drive_root: str | Path = "/content/drive/MyDrive/PCB-YOLO",
    project_root: Path | None = None,
) -> Path:
    """Copy best/last weights + metrics to Google Drive."""
    root = project_root or get_project_root()
    run_dir = run_dir_for(model_key, root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = Path(drive_root) / "runs" / model_key / f"{timestamp}_{run_dir.name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir, dest)

    print(f"Backed up to: {dest}")
    return dest


def restore_weights_from_drive(
    model_key: str,
    drive_root: str | Path = "/content/drive/MyDrive/PCB-YOLO",
    project_root: Path | None = None,
) -> Path:
    """Restore the latest backed-up run for a model into ./runs."""
    root = project_root or get_project_root()
    src_root = Path(drive_root) / "runs" / model_key
    if not src_root.exists():
        raise FileNotFoundError(f"No backups found at {src_root}")

    latest = sorted(src_root.iterdir())[-1]
    dest = run_dir_for(model_key, root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(latest, dest)
    print(f"Restored {model_key} run to: {dest}")
    return dest
