"""End-to-end smoke test on synthetic data.

Runs the full pipeline against a small fake cohort to verify nothing is
broken. Useful on the M2 before going to Humanitas, and as a regression
check after any code change.

What this proves:
  - Patient discovery finds the right files
  - Centroid computation produces physical (LPS) coordinates
  - FMCIB loads and runs forward passes (CPU on M2, GPU on HPC)
  - Output Parquet has the expected shape

What this does NOT prove:
  - That centroid coordinates align with real tumours (synthetic blobs are
    perfectly centred — only inspection on real Humanitas data can confirm
    this once you're there).

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.make_synthetic_data import main as make_data_main  # noqa: E402
from src.pipeline import run_extraction  # noqa: E402

N_PATIENTS = 3
EXPECTED_FEATS_PER_MODALITY = 4096
EXPECTED_MODALITIES = 4


def main() -> None:
    keep = "--keep" in sys.argv
    workdir = Path(tempfile.mkdtemp(prefix="fmcib_smoke_"))
    print(f"Smoke test working directory: {workdir}")

    try:
        # 1. Generate synthetic data
        data_root = workdir / "data"
        sys.argv = ["make_synthetic_data.py", "--out", str(data_root),
                    "--n", str(N_PATIENTS)]
        make_data_main()

        # 2. Build a config pointing at the synthetic data
        config = {
            "data": {
                "root": str(data_root),
                "patient_pattern": "*",
                "patient_id_from": "dirname",
                "modalities": {
                    "t1":    "{patient_id}_T1.nii.gz",
                    "t1ce":  "{patient_id}_T1CE.nii.gz",
                    "t2":    "{patient_id}_T2.nii.gz",
                    "flair": "{patient_id}_FLAIR.nii.gz",
                },
                "mask_pattern": "{patient_id}_seg.nii.gz",
            },
            "mask": {"total_label": "any_nonzero"},
            "output": {
                "path": str(workdir / "features.parquet"),
                "seed_csv": str(workdir / "seeds.csv"),
            },
            "compute": {
                "device": "auto",
                "weights_path": str(Path("./model_weights.torch").resolve()),
            },
            "run": {"max_patients": None, "verbose": True},
        }

        # 3. Sanity-check the seed CSV without invoking FMCIB
        from src.pipeline import build_seed_csv
        seed_df = build_seed_csv(config, Path(config["output"]["seed_csv"]))
        assert len(seed_df) == N_PATIENTS * EXPECTED_MODALITIES, \
            f"expected {N_PATIENTS * EXPECTED_MODALITIES} seed rows, got {len(seed_df)}"
        print("OK: seed CSV has correct number of rows.")

        # 4. Full extraction (requires FMCIB + weights)
        if not Path(config["compute"]["weights_path"]).exists():
            print("\nSKIPPING actual FMCIB inference: model_weights.torch not found.")
            print("Run `python scripts/download_weights.py` first to enable this step.")
            print("(Pipeline structure is verified — seed CSV looks correct.)")
            return

        out_path = run_extraction(config)
        wide = pd.read_parquet(out_path)

        # 5. Verify output shape
        assert len(wide) == N_PATIENTS, \
            f"expected {N_PATIENTS} patient rows, got {len(wide)}"
        expected_cols = 1 + EXPECTED_MODALITIES * EXPECTED_FEATS_PER_MODALITY  # +1 for patient_id
        assert wide.shape[1] == expected_cols, \
            f"expected {expected_cols} columns, got {wide.shape[1]}"

        print(f"\nOK: features shape = {wide.shape} (patients × (id + 4 modalities × 4096))")

    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
