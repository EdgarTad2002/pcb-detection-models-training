#!/usr/bin/env python3
"""Prepare training-only references and protected-component masks; no diffusion inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import read_dataset, audit_datasets, split_signature


def prepare(source_yaml, output, count=12, crop_size=960, seed=42, padding=8):
    if count < 1 or crop_size < 64 or padding < 0:
        raise ValueError("Require positive count, crop size >=64 and nonnegative padding")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot: {output}")
    ds = read_dataset(source_yaml)
    audit_datasets({"clean": ds}, "clean")
    rng = random.Random(seed)
    training = ds["splits"]["train"]
    # References with known capacitor centers, not unlabeled or held-out images.
    candidates = []
    for record in training:
        boxes = []
        for line in Path(record["label"]).read_text().splitlines():
            if line.strip():
                cid, cx, cy, w, h = map(float, line.split())
                boxes.append((int(cid), cx, cy, w, h))
        if any(b[0] == 0 for b in boxes):
            candidates.append((record, boxes))
    if not candidates:
        raise ValueError("Training split has no annotated capacitors")
    samples = []
    for _ in range(count * 30):
        record, normalized = rng.choice(candidates)
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
            iw, ih = image.size
            target = rng.choice([b for b in normalized if b[0] == 0])
            cw, ch = min(crop_size, iw), min(crop_size, ih)
            if min(cw, ch) < 64:
                continue
            left = max(0, min(iw - cw, int(target[1] * iw - cw / 2)))
            top = max(0, min(ih - ch, int(target[2] * ih - ch / 2)))
            crop = image.crop((left, top, left + cw, top + ch))
        mask = Image.new("L", (cw, ch), 255)
        draw = ImageDraw.Draw(mask)
        annotations = []
        for cid, cx, cy, w, h in normalized:
            original = [(cx - w / 2) * iw - left, (cy - h / 2) * ih - top,
                        (cx + w / 2) * iw - left, (cy + h / 2) * ih - top]
            x1, y1 = max(0, original[0]), max(0, original[1])
            x2, y2 = min(cw, original[2]), min(ch, original[3])
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle((max(0, math.floor(x1) - padding), max(0, math.floor(y1) - padding),
                            min(cw - 1, math.ceil(x2) + padding), min(ch - 1, math.ceil(y2) + padding)), fill=0)
            annotations.append({"class": cid, "box_xyxy": [x1, y1, x2, y2],
                                "truncated": any(abs(a - b) > 1e-6 for a, b in zip(original, (x1, y1, x2, y2)))})
        editable = sum(mask.histogram()[1:]) / (cw * ch)
        if editable < .05:
            continue
        if not samples:
            output.mkdir(parents=True)
        number = len(samples)
        reference = f"reference_{number:03d}.png"
        edit_mask = f"edit_mask_{number:03d}.png"
        crop.save(output / reference)
        mask.save(output / edit_mask)
        samples.append({"id": number, "reference": reference, "edit_mask": edit_mask,
                        "source_split": "train", "source_image": record["image"],
                        "source_sha256": record["image_sha256"], "source_label_sha256": record["label_sha256"],
                        "crop_xyxy": [left, top, left + cw, top + ch], "annotations_for_review": annotations,
                        "editable_fraction": editable,
                        "reference_sha256": hashlib.sha256((output / reference).read_bytes()).hexdigest(),
                        "mask_sha256": hashlib.sha256((output / edit_mask).read_bytes()).hexdigest(),
                        "generation_status": "not_run", "review_status": "not_reviewed"})
        if len(samples) == count:
            break
    if len(samples) != count:
        raise ValueError(f"Prepared only {len(samples)}/{count} usable references; inspect data and choose a fresh output")
    manifest = {"source_yaml": ds["yaml"], "seed": seed, "crop_size": crop_size, "padding_pixels": padding,
                "training_split_fingerprint": split_signature(training), "samples": samples,
                "mask_semantics": "white=editable background; black=protected annotated components and margins",
                "generation_status": "not_run", "contains_generated_images": False,
                "requirements_before_training": [
                    "Choose and record generator model, revision, weights and usage terms.",
                    "Use the mask for generation and explicitly composite original pixels back into all protected regions.",
                    "Reject samples that invent/remove components or invalidate annotation geometry.",
                    "Review unannotated components and truncated boxes; inherited labels are not automatically trustworthy.",
                    "Keep source boards in training only and compare against conventional augmentation with matched budgets."]}
    (output / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--crop-size", type=int, default=960)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--padding", type=int, default=8)
    args = p.parse_args()
    result = prepare(args.source_data, args.output, args.count, args.crop_size, args.seed, args.padding)
    print(f"Prepared {len(result['samples'])} training-only references and masks. No synthetic images generated.")
    print("Manifest:", args.output / "pilot_manifest.json")


if __name__ == "__main__":
    main()
