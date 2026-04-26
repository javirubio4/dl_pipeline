"""End-to-end pipeline: discover → seed CSV → FMCIB → wide-form Parquet.

Top-level entry is `run_extraction(config_dict)`. The CLI script in
`scripts/extract_features.py` just loads the YAML and calls this function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is in requirements.txt but make it optional
    def tqdm(it, **_kwargs):  # type: ignore[no-redef]
        return it

from .data import discover_patients
from .features import (
    extract_features_for_seed_csv,
    load_fmcib_model,
    select_device,
)
from .seed_points import compute_lps_centroid


def build_seed_csv(
    config: Dict[str, Any],
    out_path: Path,
) -> pd.DataFrame:
    """Discover patients and write a long-form seed CSV (one row per
    patient × modality)."""
    data_cfg = config["data"]
    mask_cfg = config.get("mask", {})
    max_patients = config.get("run", {}).get("max_patients")

    records = discover_patients(
        root=Path(data_cfg["root"]),
        patient_pattern=data_cfg["patient_pattern"],
        patient_id_from=data_cfg["patient_id_from"],
        modality_patterns=data_cfg["modalities"],
        mask_pattern=data_cfg["mask_pattern"],
    )
    if max_patients is not None:
        records = records[:max_patients]
    print(f"Discovered {len(records)} patients.", flush=True)

    rows: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="Computing centroids"):
        try:
            cx, cy, cz = compute_lps_centroid(
                record.mask, total_label=mask_cfg.get("total_label", "any_nonzero"),
            )
        except ValueError as e:
            print(f"  SKIP {record.patient_id}: {e}", flush=True)
            continue

        for modality_name, modality_path in record.modalities.items():
            rows.append({
                "patient_id": record.patient_id,
                "modality":   modality_name,
                "image_path": str(modality_path),
                "coordX":     cx,
                "coordY":     cy,
                "coordZ":     cz,
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Seed CSV written to {out_path} ({len(df)} rows).", flush=True)
    return df


def pivot_to_wide(long_features: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form (patient, modality, feat_0..feat_4095) to wide-form
    with columns `{modality}_{i}` for i in 0..4095."""
    feat_cols = [c for c in long_features.columns if c.startswith("feat_")]

    pieces = []
    for modality in sorted(long_features["modality"].unique()):
        sub = long_features[long_features["modality"] == modality].copy()
        sub = sub.set_index("patient_id")[feat_cols]
        sub.columns = [f"{modality}_{i}" for i in range(len(feat_cols))]
        pieces.append(sub)

    wide = pd.concat(pieces, axis=1).reset_index()
    return wide


def run_extraction(config: Dict[str, Any]) -> Path:
    """Full pipeline. Returns the output Parquet path."""
    out_cfg = config["output"]
    compute_cfg = config["compute"]
    run_cfg = config.get("run", {})

    seed_csv_path = Path(out_cfg["seed_csv"])
    output_path = Path(out_cfg["path"])
    verbose = run_cfg.get("verbose", False)

    # 1. Build seed CSV
    build_seed_csv(config, seed_csv_path)

    # 2. Load model
    device = select_device(compute_cfg.get("device", "auto"))
    print(f"Using device: {device}", flush=True)
    model = load_fmcib_model(Path(compute_cfg["weights_path"]), device)

    # 3. Extract
    long_features = extract_features_for_seed_csv(
        seed_csv_path, model, device, verbose=verbose
    )

    # 4. Pivot to wide form and save
    wide = pivot_to_wide(long_features)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(output_path, index=False)
    print(
        f"Wrote {len(wide)} patients × {wide.shape[1] - 1} features "
        f"to {output_path}",
        flush=True,
    )
    return output_path
