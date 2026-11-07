"""argus_uq.conformal - inductive split-CP click-disks + CRC.

Three click-disk variants:
- :func:`disk_fixed` - vanilla split-CP (Vovk 2005; A&B 2023)
- :func:`disk_normalized` - locally-weighted by plug-in UQ (Lei 2018)
- :func:`disk_cqr` - conformalised quantile regression (Romano 2019)

Two CRC variants:
- :func:`conformal_risk_control` - selective-prediction CRC (Bates 2024)
- :func:`class_conditional_crc` - per-class CRC

Plus the :func:`dispatch` entry point used by :func:`argus_uq.conformal_disk`.
"""

from argus_uq.conformal.base import (
    ClassConditionalCRCResult,
    CoordCPResult,
    CRCResult,
    point_to_bbox_distance,
)
from argus_uq.conformal.crc import (
    class_conditional_crc,
    conformal_risk_control,
)
from argus_uq.conformal.disk_cqr import disk_cqr
from argus_uq.conformal.disk_fixed import disk_fixed, inductive_cp_coords
from argus_uq.conformal.disk_normalized import disk_normalized

__all__ = [
    "CoordCPResult",
    "CRCResult",
    "ClassConditionalCRCResult",
    "point_to_bbox_distance",
    "disk_fixed",
    "disk_normalized",
    "disk_cqr",
    "inductive_cp_coords",
    "conformal_risk_control",
    "class_conditional_crc",
    "dispatch",
]


def dispatch(
    *,
    calib_clicks,
    calib_bboxes,
    test_clicks,
    test_bboxes,
    alpha: float,
    variant: str,
    calib_uq=None,
    test_uq=None,
    plug_in: str = "none",
) -> CoordCPResult:
    """Single entry point used by :func:`argus_uq.conformal_disk`."""

    if variant == "fixed":
        return disk_fixed(calib_clicks, calib_bboxes, test_clicks, test_bboxes, alpha)
    if variant == "normalized":
        if calib_uq is None or test_uq is None:
            raise ValueError("disk_normalized requires calib_uq and test_uq.")
        return disk_normalized(
            calib_clicks,
            calib_bboxes,
            calib_uq,
            test_clicks,
            test_bboxes,
            test_uq,
            alpha,
            plug_in=plug_in,
        )
    if variant == "cqr":
        if calib_uq is None or test_uq is None:
            raise ValueError("disk_cqr requires calib_uq and test_uq.")
        return disk_cqr(
            calib_clicks,
            calib_bboxes,
            calib_uq,
            test_clicks,
            test_bboxes,
            test_uq,
            alpha,
            plug_in=plug_in,
        )
    raise ValueError(f"unknown variant {variant!r}; expected fixed | normalized | cqr")
