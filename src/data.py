"""Patient and file discovery on disk.

Builds a list of `PatientRecord` objects, each containing the resolved paths to
the four MRI modalities and the segmentation mask for one patient. Designed to
fail loudly if any required file is missing — an incomplete record makes
extraction silently produce wrong results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PatientRecord:
    """All file paths for one patient."""
    patient_id: str
    modalities: Dict[str, Path]   # e.g. {"t1": Path(...), "t1ce": Path(...), ...}
    mask: Path

    def all_paths(self) -> List[Path]:
        return [*self.modalities.values(), self.mask]


def discover_patients(
    root: Optional[Path],
    patient_pattern: str,
    patient_id_from: str,
    modality_patterns: Dict[str, str],
    mask_pattern: str,
    layout: str = "per_folder",
    volumes_root: Optional[Path] = None,
    segmentations_root: Optional[Path] = None,
) -> List[PatientRecord]:
    """Find all patients and resolve their file paths.

    Args:
        root: Top-level directory for per-folder layout. Ignored for flat layout.
        patient_pattern: Glob returning one match per patient.
        patient_id_from: "dirname" or "basename" — how to derive the patient ID.
            Not used for flat layout (channel suffix is always stripped automatically).
        modality_patterns: Mapping {modality_name: filename_pattern}. The pattern
            may contain `{patient_id}` which is substituted at runtime.
        mask_pattern: Filename pattern for the segmentation mask.
        layout: "per_folder" (default, one subdirectory per patient) or "flat"
            (nnU-Net convention: volumes and segmentations in separate directories,
            channel suffix per modality file).
        volumes_root: Required for flat layout. Directory containing volume files.
        segmentations_root: Required for flat layout. Directory containing mask files.

    Returns:
        List of PatientRecord objects, sorted by patient_id.
    """
    if layout == "per_folder":
        if root is None:
            raise ValueError("root is required for layout='per_folder'")
        return _discover_per_folder(
            root=root,
            patient_pattern=patient_pattern,
            patient_id_from=patient_id_from,
            modality_patterns=modality_patterns,
            mask_pattern=mask_pattern,
        )
    if layout == "flat":
        if volumes_root is None or segmentations_root is None:
            raise ValueError(
                "volumes_root and segmentations_root are required for layout='flat'"
            )
        return _discover_flat(
            volumes_root=volumes_root,
            segmentations_root=segmentations_root,
            patient_pattern=patient_pattern,
            modality_patterns=modality_patterns,
            mask_pattern=mask_pattern,
        )
    raise ValueError(f"unknown layout: {layout!r}. Expected 'per_folder' or 'flat'")


def _discover_per_folder(
    root: Path,
    patient_pattern: str,
    patient_id_from: str,
    modality_patterns: Dict[str, str],
    mask_pattern: str,
) -> List[PatientRecord]:
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

        modalities: Dict[str, Path] = {}
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


def _discover_flat(
    volumes_root: Path,
    segmentations_root: Path,
    patient_pattern: str,
    modality_patterns: Dict[str, str],
    mask_pattern: str,
) -> List[PatientRecord]:
    """Discover patients from an nnU-Net-style flat layout.

    Volumes and segmentations live in separate directories. Each modality is a
    separate file with a channel suffix (_0000, _0001, …). The patient_pattern
    glob should match the reference-channel file (e.g. "*_0000.nii.gz") so that
    exactly one file per patient is returned.
    """
    volumes_root = Path(volumes_root).expanduser().resolve()
    segmentations_root = Path(segmentations_root).expanduser().resolve()

    if not volumes_root.is_dir():
        raise FileNotFoundError(f"volumes_root does not exist: {volumes_root}")
    if not segmentations_root.is_dir():
        raise FileNotFoundError(f"segmentations_root does not exist: {segmentations_root}")

    matches = sorted(volumes_root.glob(patient_pattern))
    if not matches:
        raise FileNotFoundError(
            f"patient_pattern {patient_pattern!r} matched nothing under {volumes_root}"
        )

    records: List[PatientRecord] = []
    for match in matches:
        # Strip .nii.gz / .nii extension, then strip trailing channel suffix (_DDDD).
        stem = match.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        patient_id = _strip_channel_suffix(stem)

        modalities: Dict[str, Path] = {}
        for name, pattern in modality_patterns.items():
            path = volumes_root / pattern.format(patient_id=patient_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"Patient {patient_id}: missing {name} modality at {path}"
                )
            modalities[name] = path

        mask_path = segmentations_root / mask_pattern.format(patient_id=patient_id)
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


def _strip_channel_suffix(stem: str) -> str:
    """Strip a trailing nnU-Net channel suffix (_DDDD) from a file stem.

    Example: "patient_001_0000" → "patient_001"
    """
    return re.sub(r"_\d{4}$", "", stem)
