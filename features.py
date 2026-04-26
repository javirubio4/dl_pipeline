"""FMCIB model wrapper for per-modality feature extraction.

Uses the lower-level FMCIB API (`fmcib_model` + `preprocess`) instead of
`get_features()` so we have explicit control over which output row maps back
to which (patient_id, modality) pair. With ~110 patients × 4 modalities = 440
forward passes per cohort, this loop runs in a few minutes on a single GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # only for static type checkers
    import torch


FEATURE_DIM = 4096  # FMCIB output dimensionality (per the paper)


def select_device(prefer: str = "auto") -> "torch.device":
    """Pick a torch device. 'auto' tries cuda → mps → cpu."""
    import torch
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


def load_fmcib_model(weights_path: Path, device: "torch.device") -> "torch.nn.Module":
    """Load FMCIB in eval mode. The package handles weight loading from
    `model_weights.torch` in the current working directory."""
    # FMCIB looks for `model_weights.torch` in CWD by default. We chdir to the
    # weights' parent so we can keep the file anywhere we like.
    import os
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(
            f"FMCIB weights not found at {weights_path}. "
            f"Run scripts/download_weights.py first."
        )
    original_cwd = Path.cwd()
    try:
        os.chdir(weights_path.parent)
        # Importing here keeps FMCIB out of the import graph until needed —
        # useful for `--help` or when running unit tests without it installed.
        from fmcib.models import fmcib_model
        model = fmcib_model()  # eval mode by default
    finally:
        os.chdir(original_cwd)

    model = model.to(device)
    model.eval()
    return model


def extract_features_for_seed_csv(
    seed_csv_path: Path,
    model: "torch.nn.Module",
    device: "torch.device",
    verbose: bool = False,
) -> pd.DataFrame:
    """Run FMCIB on every row of the seed CSV.

    The CSV must contain at least: image_path, coordX, coordY, coordZ.
    Any additional columns (e.g. patient_id, modality) are preserved and
    returned alongside the 4096-d feature vector.

    Returns a DataFrame with the original metadata columns plus
    `feat_0`…`feat_4095`.
    """
    import torch
    from fmcib.preprocessing import preprocess

    df = pd.read_csv(seed_csv_path)
    metadata_cols = [c for c in df.columns
                     if c not in {"image_path", "coordX", "coordY", "coordZ"}]

    feature_rows: List[np.ndarray] = []
    with torch.inference_mode():
        for i, row in df.iterrows():
            if verbose:
                meta = " ".join(f"{c}={row[c]}" for c in metadata_cols)
                print(f"  [{i + 1}/{len(df)}] {meta}", flush=True)

            tensor = preprocess(row)
            # `preprocess` may return a tensor without batch dim
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(device)

            out = model(tensor)
            feature_rows.append(out.squeeze(0).detach().cpu().numpy())

    feats = np.stack(feature_rows, axis=0)
    if feats.shape[1] != FEATURE_DIM:
        # Defensive — should always be 4096 but warn if FMCIB changes
        print(f"WARNING: expected {FEATURE_DIM} features, got {feats.shape[1]}")

    feat_cols = [f"feat_{i}" for i in range(feats.shape[1])]
    feat_df = pd.DataFrame(feats, columns=feat_cols)
    return pd.concat([df[metadata_cols].reset_index(drop=True), feat_df], axis=1)
