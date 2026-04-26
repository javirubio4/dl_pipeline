"""Compute the centroid of the Total tumour mask in LPS physical coordinates.

FMCIB expects seed points in LPS millimetres. SimpleITK reads NIfTI files,
converts the stored RAS affine to its internal LPS representation, and
`TransformContinuousIndexToPhysicalPoint()` therefore returns LPS directly —
no manual flipping required.

The function fails loudly if the mask contains no foreground voxels (a common
silent bug when the wrong label index is used).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import SimpleITK as sitk

LPSCoords = Tuple[float, float, float]


def compute_lps_centroid(
    mask_path: Path,
    total_label: Union[int, str] = "any_nonzero",
) -> LPSCoords:
    """Return the centroid of the Total ROI in LPS millimetres.

    Args:
        mask_path: Path to the segmentation NIfTI.
        total_label: Either an integer label value to threshold on, or the
            string "any_nonzero" to use all non-zero voxels.

    Returns:
        (x, y, z) in LPS millimetres, suitable for FMCIB's coordX/coordY/coordZ.
    """
    mask = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(mask)  # (z, y, x)

    if total_label == "any_nonzero":
        foreground = arr > 0
    elif isinstance(total_label, int):
        foreground = arr == total_label
    else:
        raise ValueError(
            f"total_label must be 'any_nonzero' or an int, got {total_label!r}"
        )

    if not foreground.any():
        raise ValueError(
            f"Mask {mask_path} has no foreground voxels matching "
            f"total_label={total_label!r}. Check Riccardo's label encoding."
        )

    # Centroid in voxel space, returned by np.where as (z, y, x)
    zyx = np.argwhere(foreground).mean(axis=0)
    # SimpleITK index order is (x, y, z)
    xyz_voxel = zyx[::-1].tolist()

    # Continuous index → physical (LPS) point in mm
    lps = mask.TransformContinuousIndexToPhysicalPoint(xyz_voxel)
    return tuple(float(c) for c in lps)  # type: ignore[return-value]
