# NOTICES

`argus-uq` builds on prior open-source work and cites the canonical
references for every UQ method and metric implemented in the package.

## Library architecture inspiration

- **lm-polygraph** (Fadeeva et al. 2024). https://github.com/IINemo/lm-polygraph.
  MIT. We adopt the StatCalculator + Estimator + UEManager design pattern.
- **torch-uncertainty** (Lafage & Laurent 2025). https://github.com/ENSTA-U2IS-AI/torch-uncertainty.
  Apache 2.0. Packaging convention (`pyproject.toml` extras) inspired this repo.

## UQ method canonical references (also cited in each estimator's docstring)

- MSP, perplexity, mean token entropy, sequence probability: standard
  logit baselines (Hendrycks & Gimpel 2017; Malinin & Gales 2018).
- Self-consistency: Wang et al. 2023 (arXiv 2203.11171).
- Semantic entropy: Kuhn et al. 2023; Farquhar et al. *Nature* 2024.
- MCSE / MCNSE: Malinin & Gales 2021.
- Lexical similarity: Fomicheva et al. 2020.
- CCP / CoCoA / CoCoA-1MCA: Vashurin et al. 2025 (AAAI).
- Mahalanobis: Lee et al. 2018.
- Mahalanobis-RMD: Ren et al. 2023 (arXiv 2209.15558).
- Mahalanobis-RDE: KernelPCA + MinCovDet (Rousseeuw 1984) on the
  positive class.
- SAPLMA: Azaria & Mitchell 2023 (EMNLP).
- SEP (Semantic Entropy Probes): Kossen et al. 2024 (arXiv 2406.15927).
- FOCUS: Zhang et al. 2023.
- RAUQ: Vazhentsev et al. 2023.
- UQAC: attention-aggregation method (2025).
- P(True): Kadavath et al. 2022 (arXiv 2207.05221).
- Verbalized 1S: Tian et al. 2023 (arXiv 2305.14975).
- Verbalized 2S: Xiong et al. 2024 (arXiv 2306.13063).
- HEDGE / IMG-HEDGE: Gautam et al. 2025.
- Inductive split CP: Vovk et al. 2005; Angelopoulos & Bates 2023.
- Conformalized quantile regression (CQR): Romano et al. 2019.
- Conformal Risk Control: Bates et al. 2024.

## Metric references

- AUROC: Hendrycks & Gimpel 2017 (incorrectness-prediction usage).
- PRR (Prediction-Rejection Ratio): Malinin & Gales 2018.
- AUSE: Ilg et al. 2018 (ECCV; per-item-error sparsification).
- ECE / MCE: Guo et al. 2017 (ICML).
- Brier score: Brier 1950.
- Selective accuracy at coverage: Geifman & El-Yaniv 2017.
- Risk-Coverage AUC: standard selective-prediction usage.

## Datasets

Licenses below are as declared by the upstream hosts.

- **ScreenSpot-v2** (Apache 2.0). https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2
- **ScreenSpot-Pro** (MIT). https://huggingface.co/datasets/likaixin/ScreenSpot-Pro
- **OSWorld-G** non-refusal subset. https://github.com/xlang-ai/OSWorld-G
  (repository: Apache 2.0; the Hugging Face copy `MMInstruction/OSWorld-G`
  declares no license).
- **UI-Vision Element Grounding** 501-item subset (MIT). https://huggingface.co/datasets/ServiceNow/ui-vision

This repository does not redistribute any screenshots. It does redistribute
item identifiers, and `data_release/closed_source_records/` also contains the
instruction text, target boxes and item metadata from ScreenSpot-v2,
ScreenSpot-Pro, OSWorld-G and UI-Vision, under the upstream terms above. To run anything
that needs images, obtain the datasets from their upstream sources.

## Closed-source APIs

The closed-source matrix uses commercial APIs from OpenAI, Anthropic,
and Google. Vendor SDK usage is governed by their respective Terms of
Service. We release request / response records; we do not redistribute
model weights.
