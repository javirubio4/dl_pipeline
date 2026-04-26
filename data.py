"""Patient and file discovery on disk.

Builds a list of `PatientRecord` objects, each containing the resolved paths to
the four MRI modalities and the segmentation mask for one patient. Designed to
fail loudly if any required file is missing — an incomplete record makes
extraction silently produce wrong results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class PatientRecord:
    """All file paths for one patient."""
    patient_id: str
    modalities: Dict[str, Path]   # e.g. {"t1": Path(...), "t1ce": Path(...), ...}
    mask: Path

    def all_paths(self) -> List[Path]:
        return [*self.modalities.values(), self.mask]


def discover_patients(
    root: Path,
    patient_pattern: str,
    patient_id_from: str,
    modality_patterns: Dict[str, str],
    mask_pattern: str,
) -> List[PatientRecord]:
    """Find all patients under `root` and resolve their file paths.

    Args:
        root: Top-level directory containing patient folders.
        patient_pattern: Glob (relative to `root`) returning one match per patient.
        patient_id_from: "dirname" or "basename" — how to derive the patient ID.
        modality_patterns: Mapping {modality_name: filename_pattern}. The pattern
            may contain `{patient_id}` which is substituted at runtime.
        mask_pattern: Filename pattern for the segmentation mask.

    Returns:
        List of PatientRecord objects, sorted by patient_id.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data.root does not exist: {root}")

    matches = sorted(root.glob(patient_pattern))
    if not matches:
        raise FileNotFoundError(
            f"patient_pattern {patient_pattern!r} matched nothing under {root}"
        )

    records: List[PatientRecord] = []
    for match in matches:
        patient_id = _patient_id_from(match, patient_id_from)
        patient_dir = match if match.is_dir() else match.parent

        modalities = {}
        for name, pattern in modality_patterns.items():
            path = patient_dir / pattern.format(patient_id=patient_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"Patient {patient_id}: missing {name} modality at {path}"
                )
            modalities[name] = path

        mask_path = patient_dir / mask_pattern.format(patient_id=patient_id)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Patient {patient_id}: missing mask at {mask_path}"
            )

        records.append(
            PatientRecord(patient_id=patient_id, modalities=modalities, mask=mask_path)
        )

    return records


def _patient_id_from(match: Path, mode: str) -> str:
    if mode == "dirname":
        return match.name
    if mode == "basename":
        # strip .nii.gz, .nii, etc.
        name = match.name
        for ext in (".nii.gz", ".nii"):
            if name.endswith(ext):
                return name[: -len(ext)]
        return match.stem
    raise ValueError(f"unknown patient_id_from mode: {mode!r}")
