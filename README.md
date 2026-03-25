# FoDiKAN
Abstract
Gene-expression classification underpins functional genomics, disease subtyping, and
biomarker discovery from transcriptomic profiles. Public microarray repositories facilitate
benchmarking, but many cohorts are high-dimensional, small-sample, and severely
imbalanced. In these regimes, multi-stage and data-adaptive pipelines can introduce information
leakage across folds, leading to over-optimistic performance estimates and less
reliable biological interpretation.
We propose FoDiKAN, a leakage-aware framework that integrates fold-local diffusion
augmentation with Kolmogorov–Arnold Networks (KANs) and is operationalized by two
algorithms: SafeCV and AugTrain. SafeCV performs leakage-safe outer cross-validation
by fitting every supervised or data-adaptive operator on the fold-local inner-training split
and keeping validation and testing real-only for epoch selection and reporting. Within
each fold, AugTrain derives a reduced space via hybrid gene selection (mRMR then
Boruta), trains a class-conditional diffusion model, and generates minority samples using
anchored DDIM sampling and geometry-aware screening. Retained synthetic samples are
used only in training with down-weighting.
Across 25 microarray datasets, we evaluate FoDiKAN under SafeCV against reference
baselines and no-synthesis KAN variants under identical fold-local preprocessing. The
best configuration achieves a macro-F1 of 88.6%, exceeding the fixed-reference Gradient
Boosting and XGBoost settings by 4.2 and 2.4 percentage points, respectively, while
remaining competitive with the balanced-reference baselines. We quantify real–synthetic
mismatch with complementary diagnostics to interpret when diffusion augmentation helps
and when it is neutral or detrimental. We also report ablation and sensitivity analyses to
assess design choices and robustness. KANs are treated as downstream backbones rather
than assumed defaults and are compared directly against a width-matched MLP under
the same protocol.
<img width="839" height="320" alt="image" src="https://github.com/user-attachments/assets/5a45f807-2937-4a93-bc49-eab09b752532" />



