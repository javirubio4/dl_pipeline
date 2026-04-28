# dl_pipeline

FMCIB feature extraction for the Humanitas meningioma MRI cohort.

Per-modality extraction (Path A): for each patient, the centroid of the Total
tumour mask is used as a seed point, and FMCIB is run independently on T1, T1CE,
T2 and FLAIR. The four 4096-d vectors are concatenated to a 16,384-d
representation per patient and saved to a Parquet file.

The classifier (RF / XGBoost / etc.) is trained downstream on a separate machine
from the saved Parquet — no GPU needed for that step.

## Project layout

```
dl_pipeline/
├── README.md
├── requirements.txt
├── configs/
│   └── humanitas_train.yaml      # adapt paths to real Humanitas layout
├── src/
│   ├── data.py                    # patient discovery, file path resolution
│   ├── seed_points.py             # mask centroid → LPS coordinates
│   ├── features.py                # FMCIB extraction wrapper
│   └── pipeline.py                # end-to-end orchestration
└── scripts/
    ├── extract_features.py        # main entry point
    ├── make_synthetic_data.py     # generate fake NIfTIs for local testing
    ├── download_weights.py        # pre-download FMCIB weights (offline HPC)
    └── smoke_test.py              # full pipeline on synthetic data
```

## Two environments

### 1. Local machine (development & smoke testing)

Verify the code runs end-to-end on
synthetic 64³ volumes, then ship the same code to the HPC.

```bash
# one-time setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# generate fake patients and run the whole pipeline on them
python scripts/smoke_test.py

# To visualize the results from the smoke test run
python scripts/smoke_test.py --keep
```

If the smoke test prints `OK: features shape = (3, 16384)` the verification worked.

### 2. Humanitas HPC (real extraction)

```bash
# clone or rsync the project to the HPC
git clone <dl_pipeline>   # or scp -r dl_pipeline/ hpc:~/

# set up env
module load python/3.10 cuda    # adjust to the cluster's modules
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# pre-download FMCIB weights once (compute nodes often have no internet)
python scripts/download_weights.py --dest ./model_weights.torch

# edit configs/humanitas_train.yaml to point at the real data
#   - data_root, modality filename patterns, mask pattern

# launch on a GPU node (wrap in your scheduler's submission file)
python scripts/extract_features.py --config configs/humanitas_train.yaml
```

The output is a single Parquet file:
`outputs/humanitas_train_features.parquet` with columns
`patient_id, t1_0…t1_4095, t1ce_0…t1ce_4095, t2_0…t2_4095, flair_0…flair_4095`.

Copy that file back to the local machine to train classifiers locally.

## Things to verify on day 1 at Humanitas

1. **Inspect one patient's files** — run `scripts/extract_features.py` with
   `max_patients: 1` and `verbose: true` in the config. Confirm the centroid
   coordinates printed look sensible (inside the brain, in mm).
2. **Check coordinate orientation** — if FMCIB seems to be cropping outside the
   tumour, the most common cause is RAS vs LPS confusion. The
   `visualize_seed_point` helper from `fmcib.visualization.verify_io` overlays
   the seed on the image; use it before running on the full cohort.
3. **Confirm modality filename convention** — update the patterns in the YAML
   config to match Riccardo's actual file naming.
4. **Confirm mask encoding** — single multi-label file or per-label files? Set
   `mask.total_label` appropriately.

Once those four things pass on a single patient, kick off the full run.
