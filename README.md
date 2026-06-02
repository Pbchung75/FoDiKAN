# Leakage-safe diffusion augmentation with KAN-based models for imbalanced microarray gene-expression classification

## Authors and Affiliations

**Bich-Chung Phan, Thanh Ma, Thanh-Nghi Do, and Huu-Hoa Nguyen**

College of Information and Communication Technology, Can Tho University, 3/2 Street, Can Tho City, 900000, Vietnam  

## Published Article

This work has been published in **Computational Biology and Chemistry**.

**DOI:** [10.1016/j.compbiolchem.2026.109057](https://doi.org/10.1016/j.compbiolchem.2026.109057)

## Abstract

Gene-expression classification underpins functional genomics, disease subtyping, and biomarker discovery from transcriptomic profiles. Public microarray repositories facilitate benchmarking, but many cohorts are high-dimensional, small-sample, and severely imbalanced. In these regimes, multi-stage and data-adaptive pipelines can introduce information leakage across folds, leading to over-optimistic performance estimates and less reliable biological interpretation.

We propose **FoDiKAN**, a leakage-aware framework that integrates fold-local diffusion augmentation with Kolmogorov–Arnold Networks (KANs) and is operationalized by two algorithms: **SafeCV** and **AugTrain**. SafeCV performs leakage-safe outer cross-validation by fitting every supervised or data-adaptive operator on the fold-local inner-training split and keeping validation and testing real-only for epoch selection and reporting. Within each fold, AugTrain derives a reduced space via hybrid gene selection, mRMR then Boruta, trains a class-conditional diffusion model, and generates minority samples using anchored DDIM sampling and geometry-aware screening. Retained synthetic samples are used only in training with down-weighting.

Across 25 microarray datasets, we evaluate FoDiKAN under SafeCV against reference baselines and no-synthesis KAN variants under identical fold-local preprocessing. The best configuration achieves a macro-F1 of 88.6%, exceeding the fixed-reference Gradient Boosting and XGBoost settings by 4.2 and 2.4 percentage points, respectively, while remaining competitive with the balanced-reference baselines. We quantify real–synthetic mismatch with complementary diagnostics to interpret when diffusion augmentation helps and when it is neutral or detrimental. We also report ablation and sensitivity analyses to assess design choices and robustness. KANs are treated as downstream backbones rather than assumed defaults and are compared directly against a width-matched MLP under the same protocol.

## Graphical Abstract

The graphical abstract summarizes the FoDiKAN workflow, including leakage-safe cross-validation, fold-local preprocessing and feature selection, class-conditional diffusion augmentation, geometry-aware filtering, and weighted downstream classification with KAN-based models.

<p align="center">
  <img src="Materials/Graphical_abstract .png" alt="Graphical abstract of the FoDiKAN framework" width="900">
</p>

<p align="center">
  <em>Graphical abstract of FoDiKAN for leakage-safe diffusion augmentation and KAN-based microarray gene-expression classification.</em>
</p>

## Highlights

- Leakage-safe outer cross-validation with real-only validation and testing.
- Fold-local preprocessing and hybrid gene selection for microarray gene expression.
- Conditional diffusion augments minority classes via anchored sampling and screening.
- Macro-F1 is 88.6% on 25 datasets, outperforming Gradient Boosting and XGBoost.
- Mismatch checks and fold-stable genes support interpretation in low-support tasks.
