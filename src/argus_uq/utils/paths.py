"""Canonical path resolution.

Every path-like string in argus_uq comes from one of:

1. A CLI argument (`argparse` / `click`).
2. An environment variable (``os.environ.get(...)``).
3. A YAML config (loaded via OmegaConf).
4. A relative path computed from this module.

No module hardcodes a user-specific path or Windows ``C:`` path.
"""

from __future__ import annotations

import os
from pathlib import Path

# Library installation root - the directory containing src/argus_uq/.
# We climb three levels: utils/paths.py -> utils/ -> argus_uq/ -> src/ -> repo
PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent
SRC_ROOT: Path = PACKAGE_ROOT.parent
REPO_ROOT: Path = SRC_ROOT.parent

# data_release/ lives at the repo root. An installed wheel has no repo root
# and uses the copy bundled at argus_uq/_data/ (see setup.py). Users can
# override both with ARGUS_DATA_DIR.
_REPO_DATA_DIR: Path = REPO_ROOT / "data_release"
_BUNDLED_DATA_DIR: Path = PACKAGE_ROOT / "_data"
DATA_RELEASE_DIR: Path = Path(
    os.environ.get(
        "ARGUS_DATA_DIR",
        _REPO_DATA_DIR if _REPO_DATA_DIR.is_dir() else _BUNDLED_DATA_DIR,
    )
)

# Per-item parquets, derived CSVs, locked subsets, closed-source records.
PER_ITEM_DIR: Path = DATA_RELEASE_DIR / "per_item_full"
DERIVED_DIR: Path = DATA_RELEASE_DIR / "derived"
SUBSETS_DIR: Path = DATA_RELEASE_DIR / "subsets"
CLOSED_SOURCE_RECORDS_DIR: Path = DATA_RELEASE_DIR / "closed_source_records"

# Optional dataset roots - set these to override HuggingFace cache fallbacks.
SCREENSPOT_PRO_DIR: Path | None = (
    Path(os.environ["SCREENSPOT_PRO_DIR"]) if "SCREENSPOT_PRO_DIR" in os.environ else None
)
SCREENSPOT_V2_DIR: Path | None = (
    Path(os.environ["SCREENSPOT_V2_DIR"]) if "SCREENSPOT_V2_DIR" in os.environ else None
)
OSWORLD_G_DIR: Path | None = (
    Path(os.environ["OSWORLD_G_DIR"]) if "OSWORLD_G_DIR" in os.environ else None
)
UI_VISION_DIR: Path | None = (
    Path(os.environ["UI_VISION_DIR"]) if "UI_VISION_DIR" in os.environ else None
)


def repo_relative(path: str | Path) -> Path:
    """Resolve `path` against `REPO_ROOT` if it is relative; else return as-is."""

    p = Path(path)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()
