# Data Card - argus-uq

Argus-UQ ships per-item uncertainty quantification scores for 28 cells
(16 open-weight + 12 closed-API) from the paper "Argus: A Cross-Regime
Study of Uncertainty Quantification Transferability in Single-Step GUI
Grounding" (NeurIPS 2026 Evaluations and Datasets Track).

## 1. Motivation

### 1.1 Purpose

Argus-UQ exists to make UQ methods comparable across vision-language
models, datasets, and access tiers under one protocol. Users can
load any cell, score any of the 27 (open-source) or 8 (closed-source)
UQ methods, and reproduce every paper number from
``data_release/derived/`` without re-running inference.

### 1.2 Funding / acknowledgement

Created by Divake Kumar, Sina Tayebati, Devashri Naik (University of
Illinois Chicago); Amanda Sofie Rios, Nilesh Ahuja, Omesh Tickoo (Intel
Labs); Ranganath Krishnan (Capital One AI Labs); Amit Ranjan Trivedi
(University of Illinois Chicago). Funding and competing interests are
stated in the paper.

## 2. Composition

### 2.1 Instances

Each instance is one (model, dataset, item) triple with:

- a stable item id
- a binary correctness label (whether the agent's greedy click landed
  in the gold bbox)
- the normalised distance from the predicted click to the bbox
- 27 (open-source) or 8 (closed-source) per-method UQ scores

There are 28 cells (model x dataset combinations) and 19,538 instances
across all cells (14,749 open-weight, 4,789 closed-API).

### 2.2 Cell catalog

**Open-source (16 cells)**: 4 models x 4 datasets

| Models | Datasets |
|---|---|
| Q7  = Qwen2.5-VL-7B-Instruct       | V2  = ScreenSpot-v2 (1,272 items) |
| Q72 = Qwen2.5-VL-72B-Instruct-AWQ  | SP  = ScreenSpot-Pro (1,581 items) |
| UI  = UI-TARS-1.5-7B               | OSG = OSWorld-G non-refusal subset (564 items) |
| PT  = POINTS-GUI-G-8B              | UIV = UI-Vision Element Grounding 501-item subset |

**Closed-source (12 cells)**: 3 vendors x 4 datasets (V2 / SP / OSG /
UIV; V2 and SP use the 300-item subsets)

| Vendors |
|---|
| OAI = OpenAI GPT-5.4 |
| Son = Anthropic Claude Sonnet 4.6 |
| Gem = Google Gemini 3.1 Pro |

### 2.3 Method panels

Open-source cells carry **27 method scores per item** spanning seven
families: logit (5), sampling (6), consistency (3), density (5),
attention (3), prompt (3), VLM-native (2).

Closed-source cells carry **8 OS-identical method scores per item**:
self_consistency, semantic_entropy, lexical_similarity, ccp,
verbalized_1s, verbalized_2s, hedge, imghedge. The other 19 methods
are open-source-only because the closed APIs do not expose token
logprobs, hidden states, or attention tensors.

## 3. Collection process

### 3.1 Open-source

- Models: HuggingFace transformers; loaded onto NVIDIA RTX A6000 Ada
  (48 GB) with bfloat16 precision (or int4-AWQ for the 72B Qwen).
- Inference: per item, run greedy decode + N=5 stochastic samples
  (temperature=1.0, top_p=0.95) + 3 prompt probes (P(True),
  Verbalized-1S, Verbalized-2S) + 3 paraphrased-instruction HEDGE
  variants + 3 image-perturbed IMG-HEDGE variants. Capture per-token
  logprobs, per-token last-layer hidden states, and per-step
  per-layer per-head attention tensors during greedy decoding.
- Scoring: Phase B reads cached records, fits trainable methods
  (SAPLMA, SEP, Mahalanobis, Mahalanobis-RMD, Mahalanobis-RDE) on a
  calibration split, and scores the 27 methods on the test split.

### 3.2 Closed-source

- API surface: vendor-isolated agents under
  ``argus_uq.models.closed_source.{openai, anthropic, google}``. The
  released classes hold configuration only; generation methods raise
  ``NotImplementedError``.
- Per-item calls: 16 (greedy + 5 samples + 3 hedge_text + 3
  hedge_image + p_true + verbalized_1s + 2 stages of verbalized_2s).
- Per-cell budget: ~300-564 items x 16 calls = ~5k-9k API calls per
  cell. ~$50-200 per cell at 2026 prices.

## 4. Preprocessing / cleaning / labelling

- Click parsing: Qwen2.5-VL tool-call output has its coordinates in
  the model-resized image space; each agent's parser rescales to
  original-image space using the smart_resize ratio.
- Closed-source vendor renames (to match the open-source method names):
  - ``coord_agreement`` -> ``ccp``
  - ``hedge_text`` -> ``hedge``
  - ``hedge_image`` -> ``imghedge``
- Closed-source ``records.jsonl`` files contain no API keys. Local
  image paths are replaced with dataset-relative paths or ``<cache>/``.
  Model responses are verbatim and can quote text visible in the public
  dataset screenshots, such as email addresses and file paths.

## 5. Uses

### 5.1 Recommended

- Reproducing the paper's headline AUROC / PRR / AUSE / ECE numbers.
- Comparing two UQ methods on a fixed cell (and confidence interval).
- Constructing deployable conformal click-disks for selective
  abstention in production grounding agents.
- Ablating UQ methods on a custom set of records (via
  ``argus_uq.manager.ScoringManager``).

### 5.2 Out of scope

- **Multi-step trajectory UQ.** Argus-UQ benchmarks per-step
  grounding only.
- **Inference-from-scratch closed-source experiments.** Vendor agents
  in this library do not implement full inference; the bundled
  closed-source records are the canonical artefact.
- **Fine-tuning models.** Argus-UQ scores frozen VLMs.

## 6. Distribution

### 6.1 License

- Library code: MIT (``LICENSE``).
- Released per-item records, derived CSVs, and subset definitions:
  CC-BY-4.0 (``LICENSE-DATA``).
- Underlying datasets carry their own upstream licenses
  (Apache 2.0 / research / MIT); see ``NOTICES.md``.

### 6.2 Where it lives

The code repository listed in ``pyproject.toml`` and ``CITATION.cff``
holds the code and all of ``data_release/``.

## 7. Maintenance

The repository is the system of record. Issues, fixes, and method
additions land via PRs.
