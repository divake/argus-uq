# argus-uq

A Python library for uncertainty quantification on vision-language-model
GUI grounding agents.

`argus-uq` is the first reproducible benchmark library that unifies
**twenty-seven UQ methods across seven families** (logit, sampling,
consistency, density, attention-trace, verbalised, VLM-native) on the
four canonical single-step grounding datasets (ScreenSpot-v2,
ScreenSpot-Pro, OSWorld-G, UI-Vision) for both **open-weight VLMs**
(Qwen2.5-VL family, UI-TARS-1.5-7B, POINTS-GUI-G-8B) and
**closed-API frontier vendors** (OpenAI GPT, Anthropic Claude, Google
Gemini, paneled across the eight vendor-API-feasible methods that share
formula identity with the open-weight panel).

It loads per-item records and 50-seed metric matrices for **16
open-weight cells and 12 closed-API cells**, re-scores any of the 27 UQ
methods on user-supplied grounding records, and emits **deployable
inductive-conformal click-disks** with formal marginal-coverage
guarantees in three variants (Disk-Fixed, Disk-Normalized, Disk-CQR).

The package accompanies the paper *Argus: A Cross-Regime Study of
Uncertainty Quantification Transferability in Single-Step GUI Grounding*
(**NeurIPS 2026 Evaluations and Datasets Track**).

## Install

Python 3.10 to 3.12.

```bash
pip install argus-uq
```

The wheel bundles the per-item scores, derived tables and subset
definitions, so `load`, `score`, `conformal_disk` and the CLIs work
without this repository. The raw closed-source API records
(`data_release/closed_source_records/`) and the master long CSVs are in
this repository only; no public call needs them.

To work from source:

```bash
git clone https://github.com/divake/argus-uq
cd argus-uq
python -m venv .venv && source .venv/bin/activate
pip install -e .                   # core library
pip install -e .[whitebox]         # + HF transformers VLM stack
pip install -e .[closed]           # + closed-source vendor SDKs
pip install -e .[paper]            # + paper-reproducing scripts
pip install -e .[dev]              # + linting and testing
pip install -e .[all]              # everything
```

To use your own copy of the data, set
`export ARGUS_DATA_DIR=/path/to/data_release`.

## Quickstart

```python
import argus_uq

# Load a cell - per-item records and per-seed scores from the released
# artifacts.
cell = argus_uq.load("PTxOSG")  # POINTS-GUI-G x OSWorld-G

# Score any of the 27 UQ methods. Returns AUROC, PRR_normalized@0.5, AUSE,
# AURC, ECE_iso, Brier_iso with bootstrap CIs (multi-seed if seeds=50).
result = argus_uq.score(cell, method="saplma", seeds=50)
print(result.auroc, result.auroc_std)   # ~0.802, ~0.011

# Build a deployable conformal click-disk at marginal coverage 1 - alpha.
disk = argus_uq.conformal_disk(
    cell,
    plug_in="saplma",             # any of the 27 UQ methods or "none"
    alpha=0.10,
    variant="normalized",         # "fixed" | "normalized" | "cqr"
)
print(disk.radius_pixels, disk.coverage)
```

## Reproducing the paper

```bash
argus_reproduce --list                     # every table name
argus_reproduce --table cells              # the 16+12 cell catalog
argus_reproduce --table leaderboard        # open-weight mean AUROC across cells
argus_reproduce --table leaderboard_closed # closed-API mean AUROC across cells
argus_reproduce --table rho                # 16x16 cross-cell Spearman matrix
argus_reproduce --all                      # every table
```

These commands print the released CSVs in `data_release/derived/`, which
hold the numbers reported in the paper. `argus_uq.score` and
`argus_uq.conformal_disk` recompute from the per-item scores with the
package's own random splits. Their AUROC means differ from the CSVs by
up to about 0.006, the top method of every closed-API cell is unchanged,
and their AUSE and conformal radii use different conventions
from the CSVs, so use the CSVs to compare against the paper.

`argus_capture` scores raw per-item signals (logprobs, hidden states,
sample clicks; see `argus_uq.records.Record`). The released parquets hold
finished scores, not raw signals, so it returns NaN or chance-level
metrics on them. Use `argus_score` for released cells.

## Layout

```
src/argus_uq/        importable library
├── api.py           the 3-entry-point Listing 1 surface
├── cell.py          Cell dataclass + per-item record loader
├── factory.py       name -> class registry
├── manager.py       ScoringManager (topo-orders stat calculators + estimators)
├── estimators/      27 UQ methods (one file per family)
├── stat_calculators/ inputs to estimators (greedy, samples, hidden, attention)
├── models/          agent abstractions (whitebox HF + closed-source APIs)
├── envs/            ScreenSpot-v2, ScreenSpot-Pro, OSWorld-G, UI-Vision
├── ue_metrics/      AUROC, PRR, AUSE, AURC, ECE, Brier, isotonic
├── conformal/       Disk-Fixed, Disk-Normalized, Disk-CQR, CRC
├── analysis/        leaderboard, cross-cell rho, family aggregates
├── data/            released-artifact loaders (parquets + CSVs)
└── utils/           paths, seeds, logging

examples/            quickstart.py and two cross-cell examples
tests/               225 unit + integration tests + 7 live smoke tests
data_release/        16+12 per-item parquets, 29 derived CSVs, 4 subsets
```

## License

- Code: MIT (`LICENSE`).
- Released datasets and per-item records: CC-BY-4.0 (`LICENSE-DATA`).
- Third-party attributions: `NOTICES.md`.

## Citation

If you use `argus-uq`, please cite the paper (`CITATION.cff` has the same
information):

```bibtex
@inproceedings{kumar2026argus,
  title     = {Argus: A Cross-Regime Study of Uncertainty Quantification
               Transferability in Single-Step GUI Grounding},
  author    = {Kumar, Divake and Tayebati, Sina and Naik, Devashri and
               Rios, Amanda Sofie and Ahuja, Nilesh and Tickoo, Omesh and
               Krishnan, Ranganath and Trivedi, Amit Ranjan},
  booktitle = {NeurIPS 2026 Evaluations and Datasets Track},
  year      = {2026}
}
```
