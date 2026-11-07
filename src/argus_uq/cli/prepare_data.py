"""argus_uq.cli.prepare_data - populate data_release/ from a research-project source.

Reads the canonical artifacts produced by the research project's
multi-seed scoring pass and copies them into ``data_release/`` inside
this package. Also rescores the closed-source cells to the 8-method
OS-identical panel from each run's panel_metrics.json.

Usage:
    python -m argus_uq.cli.prepare_data --argus-root <path>
    # or set ARGUS_ROOT in the environment:
    ARGUS_ROOT=<path> python -m argus_uq.cli.prepare_data

The ``--argus-root`` flag (or the ``ARGUS_ROOT`` env var) points at the
research project's top-level directory; this script does not embed any
specific cluster path.

This maintainer tool predates the 12-cell closed-source release: it
rebuilds the older 9-cell closed layout and should not be run over the
released ``data_release/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from argus_uq.utils.paths import DATA_RELEASE_DIR


# Map v8 run-dir name -> canonical closed-source cell id.
V8_RUN_DIR_TO_CELL = {
    "openai_ssp_300_canonical_medium": "OAIxSP",
    "openai_ssv2_300_canonical_medium_batch": "OAIxV2",
    "openai_osg_564_canonical_medium_batch": "OAIxOSG",
    "anthropic_ssp_300": "SonxSP",
    "anthropic_ssv2_300": "SonxV2",
    "anthropic_osg_564_panel_v1": "SonxOSG",
    "google_ssp_300": "GemxSP",
    "google_ssv2_300": "GemxV2",
    "google_osg_564": "GemxOSG",
}

# Map closed-source legacy column names -> canonical OS panel names.
COLUMN_RENAMES = {
    "coord_agreement": "ccp",
    "hedge_text": "hedge",
    "hedge_image": "imghedge",
}

# The 8 OS-identical methods retained on closed-source.
CLOSED_SOURCE_METHODS = (
    "self_consistency",
    "semantic_entropy",
    "lexical_similarity",
    "ccp",
    "verbalized_1s",
    "verbalized_2s",
    "hedge",
    "imghedge",
)

# Methods dropped from the closed-source panel (logprob-dependent canonical formulas).
DROPPED_LOGPROB = ("p_true", "cocoa", "cocoa_1mca")

# Methods dropped from the closed-source panel (OpenAI-only extras).
DROPPED_OPENAI_ONLY = (
    "safeground_ta",
    "safeground_ie",
    "safeground_cd",
    "safeground_com",
    "eccentricity",
    "click_bbox_entropy_13way",
    "imghedge_text_combined",
    "hedge_ensemble",
)


# Path-leak sanitiser: replaces personal home paths with placeholders.
PATH_LEAK_RE = re.compile(r"home/[a-z_][a-z0-9_]*/\.cache/")
PATH_LEAK_REPLACEMENT = "<cache>/"


def copy_open_source_parquets(argus_root: Path, out: Path) -> int:
    """Copy 16 open-source per-item parquets."""

    src_dir = argus_root / "results" / "multi_seed_n50" / "_per_item_full"
    dst_dir = out / "per_item_full"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        print(f"WARNING: {src_dir} not found; skipping open-source parquets")
        return 0
    n = 0
    for src in sorted(src_dir.glob("*.parquet")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        n += 1
    print(f"  copied {n} open-source parquets")
    return n


def copy_derived_csvs(argus_root: Path, out: Path) -> int:
    """Copy 14 derived CSVs from multi_seed_n50/derived/."""

    src_dir = argus_root / "results" / "multi_seed_n50" / "derived"
    dst_dir = out / "derived"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src_dir.is_dir():
        print(f"WARNING: {src_dir} not found; skipping derived CSVs")
        return 0
    n = 0
    for src in sorted(src_dir.glob("*.csv")):
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        n += 1
    # Also copy master_long.csv and manifest.json if present.
    for fname in ("master_long.csv", "manifest.json", "run.log"):
        src = src_dir.parent / fname
        if src.is_file():
            shutil.copy2(src, dst_dir.parent / fname)
            n += 1
    print(f"  copied {n} derived files")
    return n


def copy_subsets(argus_root: Path, out: Path) -> int:
    """Copy 4 locked subset JSONs."""

    src_dir = argus_root / "v8_closed_source" / "subsets"
    dst_dir = out / "subsets"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for canonical in ("ssv2_300.json", "ssp_300.json", "osg_564.json", "uiv_501.json"):
        src = src_dir / canonical
        if src.is_file():
            shutil.copy2(src, dst_dir / canonical)
            n += 1
    print(f"  copied {n} subset JSONs")
    return n


def rescore_closed_source_cell(
    panel_path: Path, cell_id: str, out_dir: Path
) -> int:
    """Rebuild an 8-column closed-source per-item parquet from v8 panel_metrics.json.

    Renames legacy columns to canonical OS names. Drops the
    11 non-canonical methods. Emits a parquet with: ``item_id``,
    ``correct``, ``dist`` (placeholder; closed-source per-item
    distances are not in panel_metrics), and the 8 method columns.
    """

    if not panel_path.is_file():
        print(f"WARNING: {panel_path} not found; skipping {cell_id}")
        return 0
    with open(panel_path) as fh:
        d = json.load(fh)
    per_item = d.get("per_item", [])
    if not per_item:
        print(f"WARNING: {panel_path} has no per_item; skipping {cell_id}")
        return 0

    rows: list[dict] = []
    for raw in per_item:
        item_id = raw.get("item_id", "")
        correct = int(raw.get("correct", 0))
        # Apply renames + filter to the 8 canonical methods.
        renamed = {}
        for k, v in raw.items():
            if k in ("item_id", "correct"):
                continue
            new_k = COLUMN_RENAMES.get(k, k)
            renamed[new_k] = v
        row = {"item_id": item_id, "correct": correct, "dist": float("nan")}
        for m in CLOSED_SOURCE_METHODS:
            row[m] = renamed.get(m, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{cell_id}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"  rescored {cell_id} -> {len(df)} items, 8 methods")
    return 1


def copy_closed_source_cells(argus_root: Path, out: Path) -> int:
    """Rescore + copy 9 closed-source cells."""

    runs_dir = argus_root / "v8_closed_source" / "runs"
    dst_dir = out / "per_item_full"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for run_name, cell_id in V8_RUN_DIR_TO_CELL.items():
        panel = runs_dir / run_name / "panel_metrics.json"
        n += rescore_closed_source_cell(panel, cell_id, dst_dir)

    # Drop manifest documenting what was removed.
    drops_path = out / "panel_drops.json"
    drops_path.write_text(
        json.dumps(
            {
                "logprob_dependent": list(DROPPED_LOGPROB),
                "openai_only_extras": list(DROPPED_OPENAI_ONLY),
                "renames_applied": COLUMN_RENAMES,
                "kept_methods": list(CLOSED_SOURCE_METHODS),
            },
            indent=2,
        )
    )
    return n


def sanitize_records_jsonl(src: Path, dst: Path) -> None:
    """Copy a records.jsonl while sanitising path leaks."""

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            sanitised = PATH_LEAK_RE.sub(PATH_LEAK_REPLACEMENT, line)
            fout.write(sanitised)


def copy_closed_source_records(argus_root: Path, out: Path) -> int:
    """Copy closed-source records.jsonl files (sanitised)."""

    runs_dir = argus_root / "v8_closed_source" / "runs"
    dst_dir = out / "closed_source_records"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for run_name, cell_id in V8_RUN_DIR_TO_CELL.items():
        src = runs_dir / run_name / "records.jsonl"
        if not src.is_file():
            continue
        dst = dst_dir / f"{cell_id}.records.jsonl"
        sanitize_records_jsonl(src, dst)
        n += 1
    print(f"  copied {n} sanitised records.jsonl files")
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--argus-root",
        default=os.environ.get("ARGUS_ROOT", ""),
        help="Path to the research project's top-level directory (or set ARGUS_ROOT env var).",
    )
    parser.add_argument(
        "--out",
        default=str(DATA_RELEASE_DIR),
        help="Destination directory (default: data_release/ inside the package).",
    )
    parser.add_argument(
        "--skip-records",
        action="store_true",
        help="Skip copying closed-source records.jsonl (they are large).",
    )
    args = parser.parse_args(argv)

    if not args.argus_root:
        parser.error(
            "--argus-root is required (or set ARGUS_ROOT env var). "
            "Point it at the research project's top-level directory."
        )
    argus_root = Path(args.argus_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Source: {argus_root}")
    print(f"Destination: {out}")
    print()
    print("Copying open-source parquets...")
    n_open = copy_open_source_parquets(argus_root, out)
    print()
    print("Copying derived CSVs...")
    n_der = copy_derived_csvs(argus_root, out)
    print()
    print("Copying subset JSONs...")
    n_sub = copy_subsets(argus_root, out)
    print()
    print("Rescoring closed-source cells (8-method panel)...")
    n_closed = copy_closed_source_cells(argus_root, out)
    print()
    if not args.skip_records:
        print("Copying closed-source records.jsonl (sanitised)...")
        n_recs = copy_closed_source_records(argus_root, out)
    else:
        n_recs = 0
        print("Skipping closed-source records (--skip-records).")

    print()
    print("Summary:")
    print(f"  open-source parquets: {n_open}")
    print(f"  closed-source parquets (rescored): {n_closed}")
    print(f"  derived CSVs: {n_der}")
    print(f"  subsets: {n_sub}")
    print(f"  records.jsonl: {n_recs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
