# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dl_pipeline` is a medical imaging feature extraction pipeline for the Humanitas meningioma MRI cohort. It uses FMCIB (Foundation Cancer Image Biomarker), a pre-trained deep learning model, to extract 4096-dimensional feature vectors from 3D MRI images (T1, T1CE, T2, FLAIR modalities). Output is a wide-form Parquet file with one row per patient and 16,385 columns (`patient_id` + 4 modalities × 4096 features).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

FMCIB weights (~738 MB) must be downloaded before running on HPC nodes without internet access:

```bash
python scripts/download_weights.py --dest ./model_weights.torch
```

## Common Commands

**Run the pipeline on real data:**
```bash
python scripts/extract_features.py --config configs/humanitas_train.yaml
# Override max patients for a quick test:
python scripts/extract_features.py --config configs/humanitas_train.yaml --max-patients 5
```

**End-to-end smoke test** (generates synthetic NIfTI data, runs full pipeline, validates output shape):
```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --keep   # preserve temp files for inspection
```

**Generate synthetic data for manual testing:**
```bash
python scripts/make_synthetic_data.py --out /path/to/data --n 10 --seed 42
```

## Architecture

The pipeline has four sequential stages orchestrated by `src/pipeline.py`:

1. **Patient discovery** (`src/data.py`): Scans the data root for patient folders, validates that all required modality files and the mask file exist, returns `PatientRecord` objects with resolved paths.

2. **Seed point computation** (`src/seed_points.py`): Extracts the tumor mask centroid in LPS physical coordinates using SimpleITK. The centroid is passed to FMCIB as the crop center. NIfTI files use RAS affines; SimpleITK internally converts to LPS — this conversion is intentional, not a bug.

3. **Feature extraction** (`src/features.py`): Loads the FMCIB model, runs inference on each modality image centered at the seed point, and returns a 4096-dimensional vector per (patient, modality) pair.

4. **Wide-format pivot** (`src/pipeline.py::pivot_to_wide()`): Converts the long-form DataFrame (patient × modality × features) to wide-form Parquet with columns named `{modality}_{feature_index}`.

**Data flow:** config → patient discovery → seed CSV → FMCIB inference → long DataFrame → wide Parquet

## Configuration

All pipeline parameters live in a YAML config (see `configs/humanitas_train.yaml`). Key sections:

- `data`: root path, glob patterns for patients/modalities/masks
- `mask.label`: which voxel label(s) represent the tumor (`any_nonzero` or integer)
- `output`: paths for the output Parquet and seed CSV
- `compute.device`: `auto` | `cuda` | `cpu` (MPS is explicitly disabled — Apple Silicon lacks `max_pool3d` support needed by FMCIB)
- `compute.weights_file`: path to pre-downloaded weights (omit to download at runtime)
- `run.max_patients`: limit processing to N patients (useful for testing)

## Key Constraints

- **MPS is disabled** for feature extraction (`src/features.py`) because PyTorch MPS does not support `max_pool3d`. The smoke test runs on CPU only.
- FMCIB is a heavy dependency (~2–3 GB including torch + MONAI). It downloads a 738 MB weights file from Zenodo on first use unless `weights_file` is set in the config.
- The pipeline is designed to be run on HPC clusters (SLURM) for the full cohort — pre-download weights before submitting jobs.
