#!/usr/bin/env python3
"""Read-only overlap audit before using external PCB images for training or evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.controlled_experiments import EXTENSIONS, board_id, read_dataset, split_signature


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dhash(image):
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def signatures(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixel_sha = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        versions = []
        for angle in (0, 90, 180, 270):
            rotated = image.rotate(angle, expand=True)
            versions.extend([dhash(rotated), dhash(ImageOps.mirror(rotated))])
        return {"width": image.width, "height": image.height, "pixel_sha256": pixel_sha,
                "dhash_rotations_and_mirrors": sorted(set(versions))}


def audit(candidate_dir, reference_yaml, threshold=4):
    if not 0 <= threshold <= 64:
        raise ValueError("dHash threshold must be between 0 and 64")
    candidate_dir = Path(candidate_dir).resolve()
    if not candidate_dir.is_dir():
        raise FileNotFoundError(candidate_dir)
    dataset = read_dataset(reference_yaml)
    refs = []
    for split, records in dataset["splits"].items():
        for record in records:
            refs.append({"path": record["image"], "split": split, "name": record["name"],
                         "file_sha256": record["image_sha256"], **signatures(record["image"])})
    images = sorted(p for p in candidate_dir.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if not images:
        raise ValueError("No candidate images found")
    findings = []
    for image in images:
        candidate = {"path": str(image), "file_sha256": sha256(image), **signatures(image)}
        matches = []
        for ref in refs:
            reasons = []
            if candidate["file_sha256"] == ref["file_sha256"]:
                reasons.append("identical_file")
            if candidate["pixel_sha256"] == ref["pixel_sha256"]:
                reasons.append("identical_decoded_pixels")
            if board_id(image.name) == board_id(ref["name"]):
                reasons.append("same_filename_board_identifier")
            distance = min((a ^ b).bit_count() for a in candidate["dhash_rotations_and_mirrors"]
                           for b in ref["dhash_rotations_and_mirrors"])
            if distance <= threshold:
                reasons.append("perceptual_review_candidate")
            if reasons:
                matches.append({"reference": ref["path"], "split": ref["split"],
                                "reasons": reasons, "dhash_distance": distance})
        findings.append({**candidate, "matches": matches})
    exact = lambda m: bool({"identical_file", "identical_decoded_pixels"} & set(m["reasons"]))
    return {
        "candidate_directory": str(candidate_dir), "reference_yaml": dataset["yaml"],
        "reference_split_fingerprints": {s: split_signature(rs) for s, rs in dataset["splits"].items()},
        "candidate_count": len(findings), "reference_count": len(refs), "dhash_threshold": threshold,
        "counts": {
            "candidates_with_exact_heldout_overlap": sum(any(m["split"] in {"val", "test"} and exact(m) for m in f["matches"]) for f in findings),
            "candidates_with_exact_training_overlap": sum(any(m["split"] == "train" and exact(m) for m in f["matches"]) for f in findings),
            "candidates_needing_review": sum(bool(f["matches"]) for f in findings),
        },
        "interpretation": [
            "Exact held-out matches must not be added to training.",
            "Training matches are not independent new benchmark boards.",
            "Perceptual and filename matches require visual/source-identity review; they are not proof of duplicates.",
            "No match is not proof of independence. Crops, viewpoints and physical-board reuse need further review.",
            "This tool does not verify license terms, annotation completeness, class mapping or benchmark split suitability.",
        ], "findings": findings,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-images", type=Path, required=True)
    p.add_argument("--reference-data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dhash-threshold", type=int, default=4)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit: {args.output}")
    result = audit(args.candidate_images, args.reference_data, args.dhash_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"candidate_count": result["candidate_count"], **result["counts"]}, indent=2))
    print("Saved audit:", args.output)


if __name__ == "__main__":
    main()
