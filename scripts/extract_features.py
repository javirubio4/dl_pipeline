"""CLI: python scripts/extract_features.py --config configs/humanitas_train.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as a script without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_extraction  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FMCIB features from MRIs.")
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Override config: process only first N patients "
                             "(useful for smoke tests).")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.max_patients is not None:
        config.setdefault("run", {})["max_patients"] = args.max_patients

    run_extraction(config)


if __name__ == "__main__":
    main()
