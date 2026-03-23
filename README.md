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

