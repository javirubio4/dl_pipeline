"""Pre-download FMCIB pretrained weights.

Run this on the HPC's head node (which usually has internet) before submitting
GPU jobs to compute nodes (which often don't). The default destination matches
what `src.features.load_fmcib_model` expects.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://zenodo.org/records/10528450/files/model_weights.torch?download=1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest", type=Path, default=Path("model_weights.torch"),
        help="Destination path for the downloaded weights file.",
    )
    args = parser.parse_args()

    if args.dest.exists():
        print(f"Weights already exist at {args.dest} — skipping download.")
        return

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading FMCIB weights from Zenodo to {args.dest} ...")
    urllib.request.urlretrieve(WEIGHTS_URL, args.dest)
    size_mb = args.dest.stat().st_size / 1024 / 1024
    print(f"Done. ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
