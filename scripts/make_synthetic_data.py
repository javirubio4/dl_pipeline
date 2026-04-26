"""Generate synthetic patient data for local pipeline testing.

Creates `n_patients` folders under `--out`, each containing four small
NIfTI volumes (one per modality) and a binary mask with a Gaussian "tumour"
blob in the centre. Volumes are 64³ to keep the smoke test fast.

Usage:
    python scripts/make_synthetic_data.py --out /tmp/fake_humanitas --n 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk

VOLUME_SHAPE = (64, 64, 64)        # (z, y, x)
SPACING = (1.5, 1.0, 1.0)          # mm per voxel — anisotropic on purpose
MODALITIES = ("T1", "T1CE", "T2", "FLAIR")


def make_one_patient(out_dir: Path, patient_id: str, rng: np.random.Generator) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Random tumour centre and radius
    cz, cy, cx = (rng.integers(20, 44) for _ in range(3))
    radius = rng.integers(6, 12)

    # Distance field for the spherical "tumour"
    z, y, x = np.indices(VOLUME_SHAPE)
    dist = np.sqrt((z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2)
    tumour_mask = (dist < radius).astype(np.uint8)

    for modality in MODALITIES:
        # Background tissue ~ N(50, 10), tumour patch ~ N(modality-specific, 15)
        bg = rng.normal(50, 10, VOLUME_SHAPE).astype(np.float32)
        tumour_intensity = {"T1": 80, "T1CE": 200, "T2": 150, "FLAIR": 170}[modality]
        bg[tumour_mask == 1] = rng.normal(tumour_intensity, 15, tumour_mask.sum())
        _save_nifti(bg, out_dir / f"{patient_id}_{modality}.nii.gz")

    _save_nifti(tumour_mask, out_dir / f"{patient_id}_seg.nii.gz", is_mask=True)


def _save_nifti(arr: np.ndarray, path: Path, is_mask: bool = False) -> None:
    img = sitk.GetImageFromArray(arr.astype(np.uint8 if is_mask else np.float32))
    img.SetSpacing(SPACING[::-1])  # SimpleITK expects (x, y, z)
    img.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(img, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True,
                        help="Root directory to write fake patients into.")
    parser.add_argument("--n", type=int, default=3, help="Number of patients.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        patient_id = f"P{i:03d}"
        make_one_patient(args.out / patient_id, patient_id, rng)
        print(f"  wrote {patient_id}")

    print(f"Done. {args.n} fake patients under {args.out}")


if __name__ == "__main__":
    main()
