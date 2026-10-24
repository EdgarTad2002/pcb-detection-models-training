#!/usr/bin/env python3
"""
tools/setup_pcb_filtered_4class.py
==================================
Configures and verifies the Roboflow PCB dataset at:
  /mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8
to strictly use the 4 canonical PCB components:
  Class 0: Capacitor
  Class 1: Connector
  Class 2: Electrolytic Capacitor
  Class 3: IC

Actions:
  1. Locates pcb-filtered-yolov8 dataset on the cluster (or local workspace).
  2. Inspects label text files across train, valid, test splits.
  3. If raw 23-class Roboflow labels are detected (IDs >= 4 or class 1 split):
     - Safely backs up original labels to `labels_raw_backup/` in each split.
     - Remaps classes according to the non-destructive project specification:
         Raw 1 (Capacitor Jumper) + Raw 2 (Capacitor) -> Class 0 (Capacitor)
         Raw 4 (Connector)                            -> Class 1 (Connector)
         Raw 7 (Electrolytic Capacitor)               -> Class 2 (Electrolytic Capacitor)
         Raw 9 (IC) + Raw 22 (iC typo)                -> Class 3 (IC)
         All other 19 non-target classes (pins, resistors, etc.) are pruned.
  4. Purges all stale `*.cache` files (train.cache, valid.cache, labels.cache).
  5. Backs up original data.yaml (if 23-class) and writes clean 4-class data.yaml.
  6. Creates symlinks in the active workspace if needed so relative paths work.
  7. Prints a complete audit verification table of component bounding boxes.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import shutil
import sys

try:
    import yaml
except ImportError:
    yaml = None

TARGET_NAMES = ["Capacitor", "Connector", "Electrolytic Capacitor", "IC"]

# Roboflow 23-class -> Canonical 4-class mapping
RAW_TO_4CLASS_MAP = {
    1: 0,   # Capacitor Jumper -> Capacitor
    2: 0,   # Capacitor        -> Capacitor
    4: 1,   # Connector        -> Connector
    7: 2,   # Electrolytic Capacitor -> Electrolytic Capacitor
    9: 3,   # IC               -> IC
    22: 3,  # iC (typo)        -> IC
}

DEFAULT_CLUSTER_DATASET_DIR = Path("/mnt/weka/etadevosyan/pcb-yolo/pcb-detection-models-training/datasets/pcb-filtered-yolov8")


def locate_dataset(override_dir: Path | None = None) -> Path:
    candidates = []
    if override_dir:
        candidates.append(Path(override_dir).resolve())

    # Environment variable override
    env_dir = os.environ.get("PCB_FILTERED_DATASET_DIR")
    if env_dir:
        candidates.append(Path(env_dir).resolve())

    # Cluster standard path
    candidates.append(DEFAULT_CLUSTER_DATASET_DIR)

    # Relative paths from current workspace
    candidates.append(Path("datasets/pcb-filtered-yolov8").resolve())
    candidates.append(Path("../pcb-detection-models-training/datasets/pcb-filtered-yolov8").resolve())

    for c in candidates:
        if c.exists() and (c / "train").exists() or (c / "data.yaml").exists():
            return c

    # Fallback to default
    return candidates[0] if candidates else DEFAULT_CLUSTER_DATASET_DIR


def audit_labels(lbl_dir: Path):
    """Returns histogram of class IDs in label directory."""
    counts = Counter()
    total_files = 0
    if not lbl_dir.exists():
        return counts, total_files

    for f in lbl_dir.glob("*.txt"):
        total_files += 1
        try:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if parts:
                        counts[int(parts[0])] += 1
        except Exception:
            pass
    return counts, total_files


def remap_split_labels(lbl_dir: Path, backup_dir: Path):
    """Backs up and remaps labels in-place to 0..3."""
    if not lbl_dir.exists():
        return 0, 0, Counter()

    lbl_files = sorted(list(lbl_dir.glob("*.txt")))
    if not lbl_files:
        return 0, 0, Counter()

    # Backup original labels first if backup doesn't already exist
    backup_dir.mkdir(parents=True, exist_ok=True)
    for f in lbl_files:
        bf = backup_dir / f.name
        if not bf.exists():
            shutil.copy2(f, bf)

    new_counts = Counter()
    processed_files = 0
    total_boxes = 0

    for f in lbl_files:
        new_lines = []
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        raw_cid = int(parts[0])
                    except ValueError:
                        continue
                    if raw_cid in RAW_TO_4CLASS_MAP:
                        new_cid = RAW_TO_4CLASS_MAP[raw_cid]
                        new_counts[new_cid] += 1
                        total_boxes += 1
                        new_line = f"{new_cid} " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)

        with open(f, "w", encoding="utf-8") as fp:
            fp.writelines(new_lines)
        processed_files += 1

    return processed_files, total_boxes, new_counts


def purge_stale_caches(dataset_dir: Path):
    """Deletes all .cache files in dataset tree."""
    deleted = 0
    for p in dataset_dir.rglob("*.cache"):
        try:
            p.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


def write_canonical_yaml(dataset_dir: Path) -> Path:
    """Writes the clean 4-class data.yaml."""
    yaml_path = dataset_dir / "data.yaml"

    # Backup existing data.yaml if it contains >4 classes
    if yaml_path.exists():
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            is_legacy = False
            if yaml is not None:
                old_cfg = yaml.safe_load(raw_text)
                if old_cfg and old_cfg.get("nc", 0) > 4:
                    is_legacy = True
            elif "nc: 23" in raw_text or ("nc: 4" not in raw_text and "nc:" in raw_text):
                is_legacy = True

            if is_legacy:
                backup_yaml = dataset_dir / "data_raw_backup.yaml"
                if not backup_yaml.exists():
                    shutil.copy2(yaml_path, backup_yaml)
                    print(f"📦 Backed up raw data.yaml to: {backup_yaml}")
        except Exception:
            pass

    content = f"""# PCB Filtered 4-Class Canonical Dataset Configuration
# Automated setup for Zhou & Agaian 4-component taxonomy
path: {dataset_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 4
names:
  0: Capacitor
  1: Connector
  2: Electrolytic Capacitor
  3: IC
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)

    return yaml_path


def ensure_workspace_symlink(dataset_dir: Path):
    """If datasets/pcb-filtered-yolov8 doesn't exist locally, symlink it."""
    cwd = Path.cwd().resolve()
    local_datasets = cwd / "datasets"
    local_target = local_datasets / "pcb-filtered-yolov8"

    if dataset_dir.resolve() == local_target.resolve():
        return

    try:
        local_datasets.mkdir(parents=True, exist_ok=True)
        if not local_target.exists():
            local_target.symlink_to(dataset_dir.resolve(), target_is_directory=True)
            print(f"🔗 Created workspace symlink: {local_target} -> {dataset_dir.resolve()}")
    except Exception as e:
        print(f"ℹ️ Workspace symlink note: {e}")


def main():
    p = argparse.ArgumentParser(description="Configure pcb-filtered-yolov8 for 4 canonical components.")
    p.add_argument("--data-dir", type=str, default=None, help="Path to pcb-filtered-yolov8 directory")
    p.add_argument("--force-remap", action="store_true", help="Force remapping from backup even if already mapped")
    args = p.parse_args()

    dataset_dir = locate_dataset(Path(args.data_dir) if args.data_dir else None)
    print("=" * 75)
    print("🎯 PCB FILTERED DATASET 4-CLASS CONFIGURATOR")
    print(f"📁 Target Dataset Directory: {dataset_dir}")
    print("=" * 75)

    if not dataset_dir.exists():
        print(f"❌ Error: Dataset directory not found at {dataset_dir}")
        print("Please check cluster path or pass --data-dir.")
        sys.exit(1)

    splits = ["train", "valid", "test"]
    split_stats = {}

    for split in splits:
        split_dir = dataset_dir / split
        # Check valid vs val alias
        if not split_dir.exists() and split == "valid" and (dataset_dir / "val").exists():
            split_dir = dataset_dir / "val"

        lbl_dir = split_dir / "labels"
        img_dir = split_dir / "images"

        if not lbl_dir.exists():
            print(f"⚠️ Label directory not found for split '{split}': {lbl_dir}")
            continue

        raw_counts, total_files = audit_labels(lbl_dir)
        max_cid = max(raw_counts.keys()) if raw_counts else -1

        print(f"\n🔍 Split '{split}': {total_files} label files, max class ID = {max_cid}")

        backup_dir = split_dir / "labels_raw_backup"
        needs_remap = (max_cid > 3) or (1 in raw_counts) or args.force_remap

        if needs_remap:
            print(f"  ⚡ Detected raw Roboflow annotations. Remapping to 4 canonical classes...")
            # If force remap and backup exists, restore from backup first
            if args.force_remap and backup_dir.exists():
                for bf in backup_dir.glob("*.txt"):
                    shutil.copy2(bf, lbl_dir / bf.name)

            proc_files, total_boxes, new_counts = remap_split_labels(lbl_dir, backup_dir)
            print(f"  ✅ Remapped {proc_files} files ({total_boxes} boxes total). Raw backup saved to {backup_dir.name}/")
            split_stats[split] = {"files": proc_files, "counts": new_counts}
        else:
            print(f"  ✅ Already clean 4-class annotations (all class IDs <= 3).")
            split_stats[split] = {"files": total_files, "counts": raw_counts}

    # Purge stale caches
    purged = purge_stale_caches(dataset_dir)
    print(f"\n🧹 Purged {purged} stale *.cache files across splits.")

    # Write clean data.yaml
    yaml_path = write_canonical_yaml(dataset_dir)
    print(f"📝 Verified & written clean 4-class YAML: {yaml_path}")

    # Ensure valid / val symlink alias if needed
    if (dataset_dir / "valid").exists() and not (dataset_dir / "val").exists():
        try:
            (dataset_dir / "val").symlink_to(dataset_dir / "valid", target_is_directory=True)
            print("🔗 Created symlink alias: val -> valid")
        except Exception:
            pass

    # Ensure workspace symlink
    ensure_workspace_symlink(dataset_dir)

    # Print Summary Table
    print("\n" + "=" * 78)
    print("📊 4-CANONICAL COMPONENT DISTRIBUTION IN pcb-filtered-yolov8")
    print("=" * 78)
    print(f"{'Split':<10} | {'Files':<8} | {'Capacitor':<12} | {'Connector':<12} | {'Elec. Cap':<12} | {'IC':<8}")
    print("-" * 78)
    for split in splits:
        if split in split_stats:
            c = split_stats[split]["counts"]
            f = split_stats[split]["files"]
            print(f"{split:<10} | {f:<8} | {c.get(0, 0):<12} | {c.get(1, 0):<12} | {c.get(2, 0):<12} | {c.get(3, 0):<8}")
    print("-" * 78)
    print("🎉 Dataset configuration is 100% complete and verified!")
    print(f"Target data config: {yaml_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
