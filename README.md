# FoDiKAN

FoDiKAN is a modularized refactor of the original experiment script for fold-local feature selection, diffusion-based minority augmentation, SMOTE-family baselines, and KAN backbones on gene-expression datasets.

## Structure

```text
FoDiKAN/
├── scripts/
│   └── run_experiment.py
├── src/
│   └── fodikan/
│       ├── config.py
│       ├── pipeline.py
│       ├── data/
│       ├── diffusion/
│       ├── evaluation/
│       ├── features/
│       ├── models/
│       ├── results/
│       ├── training/
│       └── utils/
├── requirements.txt
├── requirements-core.txt
├── requirements-kan.txt
└── pyproject.toml
```

## Main points

- The main pipeline comes from `Fodiffkan_full_with_smote_family_patched.py`.
- The MLP utilities were extracted from `Fodiffkan_full_backbonefair.py` and placed in `src/fodikan/models/mlp.py`.
- Absolute local paths were removed and replaced with public-friendly defaults:
  - `data/`
  - `results/`
- The undefined `lambda_js_eff` / `lambda_mmd_eff` metadata references inside the SMOTE-family branch were removed.

## Installation

Core dependencies only:

```bash
pip install -r requirements-core.txt
```

Full pipeline, including KAN backbones:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python scripts/run_experiment.py --data_path data --results_path results
```

To restrict the run to classical baselines only:

```bash
python scripts/run_experiment.py --models SVM RF GB XGB --modes NoAug
```

## Notes

- `src/fodikan/models/mlp.py` is kept as an optional utility module. It is not wired into the default paper-aligned pipeline.
- The default CLI still follows the original main script and runs the KAN backbones unless you override `--models`.
