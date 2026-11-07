# data_release/

This directory holds the released artifacts. The Python wheel bundles
`per_item_full/`, `derived/`, `subsets/`, `manifest.json` and
`panel_drops.json`. `closed_source_records/` and the `master_long*.csv` files
are in this repository only.

## Layout

```
data_release/
├── per_item_full/      28 parquets, one per cell (16 open-weight + 12 closed-API)
├── derived/            29 canonical CSVs that drive the paper tables
├── subsets/            4 locked stratified subset definitions (300/300/564/501)
├── closed_source_records/
                        Per-cell JSONLs, 12 files (no API keys)
├── master_long.csv     per-seed metrics, open-weight cells
├── master_long_closed.csv
│                       per-seed metrics, closed-API cells
├── manifest.json       open-weight run manifest
└── panel_drops.json    methods excluded from the closed-API panel
```

## Contents

### `per_item_full/`

Per-item per-method UQ scores. Schema:

- `item_id` (str): stable identifier
- `correct` (int 0/1): point-in-bbox correctness label
- `dist` (float): normalised distance from predicted click to gold bbox
  (relative to the image diagonal)
- **27 method columns** for open-weight cells (Q7xV2, Q7xSP, Q7xOSG,
  Q7xUIV, Q72xV2, Q72xSP, Q72xOSG, Q72xUIV, UIxV2, UIxSP, UIxOSG,
  UIxUIV, PTxV2, PTxSP, PTxOSG, PxUIV): one column per UQ method,
  higher value = more uncertain
- Closed-API cells (OAIxV2, OAIxSP, OAIxOSG, OAIxUIV, SonxV2, SonxSP,
  SonxOSG, SonxUIV, GemxV2, GemxSP, GemxOSG, GemxUIV) instead store
  `greedy_x`, `greedy_y`, `gold_x`, `gold_y` and `d_pix` (click-to-gold
  distance in pixels). `image_w` and `image_h` are unused (0).
  `argus_uq.load` maps these to the common schema, with `dist` set to
  `d_pix` over a per-dataset representative image diagonal.
- **8 method columns** for closed-API cells: the eight
  vendor-API-feasible methods with formula identity to the
  open-weight panel: `self_consistency`, `semantic_entropy`,
  `lexical_similarity`, `ccp`, `verbalized_1s`, `verbalized_2s`,
  `hedge`, `imghedge`

### `derived/`

Pre-computed paper tables produced by 50 random calibration/test
splits over the per-item parquets. The files below cover the open-weight
cells; each has a `*_closed.csv` counterpart for the closed-API cells, and
`cross_open_closed_rho.csv` compares the two panels.

Closed-API CSVs and `master_long_closed.csv` use the run names below as
`cell_id`:

| `cell_id` in CSVs | Cell | | `cell_id` in CSVs | Cell |
|---|---|---|---|---|
| `openai_v2_300` | OAIxV2 | | `anthropic_sp` | SonxSP |
| `openai_sp_300` | OAIxSP | | `anthropic_osg` | SonxOSG |
| `openai_osg` | OAIxOSG | | `anthropic_uiv` | SonxUIV |
| `openai_uiv` | OAIxUIV | | `gemini_v2_300` | GemxV2 |
| `anthropic_v2` | SonxV2 | | `gemini_sp` | GemxSP |
| `gemini_osg` | GemxOSG | | `gemini_uiv` | GemxUIV |


- `cell_winners.csv`: per-cell top-1 method by AUROC
- `family_best_per_cell.csv`: best method per family per cell
- `method_leaderboard.csv`: global method ranking
- `family_means.csv`: per-family aggregate means
- `cross_cell_rho.csv`, `cross_cell_rho_pairs.csv`,
  `cross_cell_rho_stats.csv`: Spearman ρ matrix of method rankings
  across the 16 open-weight cells
- `ause_winners.csv`: AUROC top-1 vs AUSE top-1 per cell
- `tuning_deltas.csv`: Q7 → UI fine-tuning deltas per method on SP
- `bootstrap_ci.csv`: 500-resample bootstrap CIs
- `significance_pairs.csv`, `nan_audit.csv`,
  `conformal_disks.csv`, `conformal_disks_summary.csv`

### `subsets/`

- `ssv2_300.json`, `ssp_300.json`: 300-item stratified subsets used
  by the closed-API matrix (seed=42)
- `osg_564.json`: full OSWorld-G non-refusal set
- `uiv_501.json`: UI-Vision 501-item curated subset

### `closed_source_records/`

Per-call JSONL records from the OpenAI / Anthropic / Google API
panels, one file per cell, with the raw vendor request / response
bodies for audit. They contain no API keys and no screenshots. Model responses can quote text
visible in the public dataset screenshots, including email addresses
and file paths shown in those images.

## Loading

```python
import argus_uq
cell = argus_uq.load("PTxOSG")
print(cell.records.head())
```

The `argus_uq.data` subpackage handles parquet / CSV / JSON parsing.
