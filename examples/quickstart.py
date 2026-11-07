"""The canonical argus-uq quickstart.

``tests/test_listing1.py`` asserts that this script runs end-to-end.
"""

import argus_uq

# Load a cell - per-item records and per-seed scores from data_release/.
cell = argus_uq.load("PTxOSG")  # POINTS-GUI-G x OSWorld-G

# Score any of the 27 UQ methods. Returns AUROC, PRR_normalized@0.5, AUSE,
# AURC, ECE_iso, Brier_iso with bootstrap CIs (multi-seed if seeds=50).
result = argus_uq.score(cell, method="saplma", seeds=50)
print(f"AUROC: {result.auroc:.3f} +/- {result.auroc_std:.3f}")

# Build a deployable conformal click-disk at marginal coverage 1 - alpha.
disk = argus_uq.conformal_disk(
    cell,
    plug_in="saplma",  # any of the 27 UQ methods or "none"
    alpha=0.10,
    variant="normalized",  # "fixed" | "normalized" | "cqr"
)
print(f"Disk radius: {disk.radius_pixels:.0f} px (coverage {disk.coverage:.3f})")
