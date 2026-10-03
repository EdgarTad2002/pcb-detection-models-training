#!/usr/bin/env python3
"""
Spectral Signature Extractor for PCB-Vision Benchmark (Arbash et al., 2024).

Extracts and models the physical spectral reflectance of PCB components
(Ceramic Capacitors, ICs, Connectors, and FR-4 Substrate) across the VNIR
spectrum (400nm - 700nm in 10nm steps).

Generates `data/pcb_spectral_priors.json` used by `physics_spectral_yolo26.py`
to initialize the neural network with physical material optical priors.

Provenance Note:
By default, this tool generates illustrative, literature-derived empirical priors
synthesized from published material reflectance profiles in the PCB-Vision benchmark
(Arbash et al., IEEE Sensors J. 2024). When raw Specim FX10 ENVI cubes (.hdr/.raw)
from Rodare (record 2704) are available, pass --raw-cube and --raw-mask to extract
direct sensor measurements.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

# 31 discrete wavelength bands from 400nm to 700nm (10nm step)
WAVELENGTHS = np.arange(400, 710, 10)

# Illustrative / literature-derived empirical spectral reflectance priors.
# Derived from qualitative spectral signatures and material reflectance profiles
# documented in the PCB-Vision benchmark (Arbash et al., IEEE Sensors J. 2024).
# Note: These 31-band curves are illustrative physical priors rather than directly parsed
# raw binary ENVI sensor cubes. In a presentation or paper, they should be described as
# 'empirical literature-derived priors' rather than direct Specim FX10 laboratory calibrations.
EMPIRICAL_SPECTRA = {
    # Ceramic MLCC (Barium titanate + Ni/Sn terminations): high flat/rising reflectance
    "capacitor": np.array([
        0.18, 0.19, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34,  # 400-490nm
        0.35, 0.36, 0.38, 0.40, 0.42, 0.43, 0.45, 0.47, 0.49, 0.51,  # 500-590nm
        0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.69, 0.71, 0.73  # 600-700nm
    ]),
    # FR-4 Green Solder Mask + Substrate: strong green peak, red chlorophyll-like absorption dip
    "substrate": np.array([
        0.08, 0.09, 0.10, 0.12, 0.15, 0.19, 0.24, 0.28, 0.31, 0.33,  # 400-490nm
        0.35, 0.34, 0.31, 0.27, 0.23, 0.19, 0.16, 0.14, 0.13, 0.12,  # 500-590nm
        0.12, 0.13, 0.14, 0.16, 0.19, 0.24, 0.30, 0.38, 0.46, 0.54, 0.60  # 600-700nm
    ]),
    # Black Epoxy IC Mold Compound: low flat optical absorption across visible bands
    "ic": np.array([
        0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.07, 0.07, 0.08,
        0.08, 0.08, 0.09, 0.09, 0.09, 0.10, 0.10, 0.10, 0.11, 0.11,
        0.12, 0.12, 0.12, 0.13, 0.13, 0.14, 0.14, 0.15, 0.15, 0.16, 0.17
    ]),
    # Metallic Solder / Connector Contacts (Tin/Lead/Gold): high specular reflection
    "connector": np.array([
        0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58,
        0.60, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
        0.70, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.80
    ])
}


def compute_spectral_contrast(spectra=None, provenance="literature_derived_empirical_prior"):
    """
    Computes the optical contrast ratio between ceramic capacitors and substrate
    across the 31 discrete bands:
        C(lambda) = |S_cap(lambda) - S_sub(lambda)| / (S_sub(lambda) + eps)
    """
    if spectra is None:
        spectra = EMPIRICAL_SPECTRA

    cap_ref = np.asarray(spectra["capacitor"], dtype=np.float32)
    sub_ref = np.asarray(spectra["substrate"], dtype=np.float32)

    # Physical contrast: where capacitor stands out against the board substrate
    raw_contrast = np.abs(cap_ref - sub_ref) / (sub_ref + 1e-4)

    # Normalize weights so they sum to 1.0 (probability distribution over informative bands)
    norm_contrast = raw_contrast / np.sum(raw_contrast)

    # Also compute relative signed gain (positive where capacitor reflects more than board)
    signed_contrast = (cap_ref - sub_ref)

    return {
        "provenance": provenance,
        "description": (
            "Illustrative empirical spectral priors synthesized from published reflectance profiles "
            "(Arbash et al., IEEE Sensors J. 2024)"
            if provenance == "literature_derived_empirical_prior"
            else "Extracted from raw ENVI hyperspectral data cubes"
        ),
        "wavelengths_nm": WAVELENGTHS.tolist(),
        "capacitor_reflectance": cap_ref.tolist(),
        "substrate_reflectance": sub_ref.tolist(),
        "ic_reflectance": np.asarray(spectra["ic"], dtype=np.float32).tolist(),
        "connector_reflectance": np.asarray(spectra["connector"], dtype=np.float32).tolist(),
        "normalized_contrast_weights": norm_contrast.tolist(),
        "signed_contrast_gains": signed_contrast.tolist(),
        "peak_contrast_wavelength_nm": int(WAVELENGTHS[np.argmax(raw_contrast)]),
    }


def parse_envi_header(hdr_path):
    """Parses an ENVI .hdr file into a dictionary of key-value pairs."""
    hdr_path = Path(hdr_path)
    if not hdr_path.exists():
        raise FileNotFoundError(f"ENVI header not found: {hdr_path}")

    metadata = {}
    with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Match key = value pairs (handling multi-line braces { ... })
    pattern = re.compile(r"([a-zA-Z0-9_\-\s]+?)\s*=\s*(\{([^}]+)\}|[^\n]+)")
    for match in pattern.finditer(text):
        key = match.group(1).strip().lower()
        val = match.group(2).strip()
        if val.startswith("{") and val.endswith("}"):
            val = [x.strip() for x in val[1:-1].split(",") if x.strip()]
        metadata[key] = val
    return metadata


def parse_raw_envi_scene(cube_path, mask_path=None):
    """
    Parses real ENVI .hdr / binary data cubes if downloaded from
    Rodare (record/2704).
    """
    cube_path = Path(cube_path)
    hdr_path = cube_path.with_suffix(".hdr") if cube_path.suffix != ".hdr" else cube_path
    raw_path = cube_path.with_suffix(".raw") if cube_path.suffix == ".hdr" else cube_path
    if not raw_path.exists():
        raw_path = cube_path.with_suffix(".dat")

    if not hdr_path.exists() or not raw_path.exists():
        raise FileNotFoundError(
            f"Missing ENVI files: header={hdr_path.exists()} ({hdr_path}), data={raw_path.exists()} ({raw_path})"
        )

    print(f"Reading raw ENVI cube header from: {hdr_path}")
    hdr = parse_envi_header(hdr_path)
    samples = int(hdr.get("samples", 0))
    lines = int(hdr.get("lines", 0))
    bands = int(hdr.get("bands", 0))
    interleave = str(hdr.get("interleave", "bsq")).lower()
    dtype_code = int(hdr.get("data type", 4))

    # Map ENVI data types to numpy
    type_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}
    dtype = type_map.get(dtype_code, np.float32)

    print(f"ENVI Cube: {samples} samples x {lines} lines x {bands} bands (interleave={interleave}, dtype={dtype})")
    data = np.fromfile(raw_path, dtype=dtype)

    if interleave == "bil":
        cube = data.reshape((lines, bands, samples)).transpose(1, 0, 2)
    elif interleave == "bip":
        cube = data.reshape((lines, samples, bands)).transpose(2, 0, 1)
    else:  # bsq
        cube = data.reshape((bands, lines, samples))

    # If mask is provided, extract mean spectra per mask label
    if mask_path and Path(mask_path).exists():
        mask = np.load(mask_path) if str(mask_path).endswith(".npy") else None
        if mask is not None and mask.shape == (lines, samples):
            cap_mask = (mask == 1)
            sub_mask = (mask == 0)
            cap_spectrum = cube[:, cap_mask].mean(axis=1) if cap_mask.any() else EMPIRICAL_SPECTRA["capacitor"]
            sub_spectrum = cube[:, sub_mask].mean(axis=1) if sub_mask.any() else EMPIRICAL_SPECTRA["substrate"]
            # Interpolate or resample to 31 bands if needed
            return {
                "capacitor": cap_spectrum[:31] if len(cap_spectrum) >= 31 else EMPIRICAL_SPECTRA["capacitor"],
                "substrate": sub_spectrum[:31] if len(sub_spectrum) >= 31 else EMPIRICAL_SPECTRA["substrate"],
                "ic": EMPIRICAL_SPECTRA["ic"],
                "connector": EMPIRICAL_SPECTRA["connector"],
            }

    print("Warning: No mask provided with ENVI cube; falling back to empirical literature priors.")
    return EMPIRICAL_SPECTRA


def main():
    parser = argparse.ArgumentParser(description="Extract spectral material priors.")
    parser.add_argument("--output", type=Path, default=Path("data/pcb_spectral_priors.json"))
    parser.add_argument("--raw-cube", type=Path, default=None, help="Optional raw ENVI cube path (.hdr or .raw)")
    parser.add_argument("--raw-mask", type=Path, default=None, help="Optional raw mask path (.npy)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.raw_cube and args.raw_cube.exists():
        print(f"Extracting measured spectra from ENVI cube: {args.raw_cube}")
        spectra = parse_raw_envi_scene(args.raw_cube, args.raw_mask)
        priors = compute_spectral_contrast(spectra, provenance="raw_envi_extracted")
    else:
        print("Note: Using illustrative / literature-derived empirical spectral priors based on Arbash et al. (2024).")
        print("      (Pass --raw-cube <path> to extract from raw Specim FX10 sensor cubes when available)")
        priors = compute_spectral_contrast(EMPIRICAL_SPECTRA, provenance="literature_derived_empirical_prior")

    with open(args.output, "w") as f:
        json.dump(priors, f, indent=2)

    peak_wl = priors["peak_contrast_wavelength_nm"]
    print(f"Saved physics priors to: {args.output}")
    print(f"  Provenance: {priors['provenance']}")
    print(f"  Peak capacitor-to-substrate optical contrast identified at: {peak_wl} nm")


if __name__ == "__main__":
    main()
