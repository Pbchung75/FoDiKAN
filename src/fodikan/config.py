from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

DEFAULT_DATA_DIR = os.getenv("FODIKAN_DATA_DIR", "data")
BASE_RESULTS_PATH = os.getenv("FODIKAN_RESULTS_DIR", "results")

@dataclass
class Args:
    data_path: str
    results_path: str
    run_name: str
    dataset_ids: Optional[List[str]]

    models: List[str]
    modes: List[str]

    smote_k_neighbors: int
    smote_m_neighbors: int
    borderline_kind: str
    kmeans_n_clusters: int

    extra_python_path: List[str]

    seed: int
    val_ratio: float

    batch_size: int
    num_epochs: int
    patience: int

    no_fs: bool
    no_diffusion: bool
    no_hdbscan: bool
    no_anchor: bool
    no_knn: bool

    gamma: float
    w_syn: float
    keep_q: float
    knn_k: int
    min_real_for_synthesis: int

    save_synthesis: bool
    save_fold_artifacts: bool

    pre_screen_topk: Optional[int]
    mrmr_candidate_pool: Optional[int]
    mrmr_topk: int
    mrmr_redundancy_threshold: float
    boruta_n_estimators: int
    boruta_max_iter: int
    boruta_alpha: float
    fs_min_keep: int
    fs_fallback_topk: int

    alignment_bins: int
    diff_lambda_js: float
    diff_lambda_mmd: float

    log_file: str

def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Paper-aligned FoDiKAN implementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--data_path",
        default=DEFAULT_DATA_DIR,
        help=(
            "Root folder containing datasets. Supported structure: data_root/<dataset_id>/data.trn.gz. "
            "Defaults to DEFAULT_DATA_DIR."
        ),
    )
    p.add_argument(
        "--results_path",
        default=BASE_RESULTS_PATH,
        help="Folder where results are written. Defaults to BASE_RESULTS_PATH.",
    )
    p.add_argument("--run_name", default="FoDiKAN_paper_aligned", help="Run subfolder name.")
    p.add_argument("--dataset_ids", nargs="*", default=None, help="Optional subset of dataset IDs.")

    p.add_argument(
        "--models",
        nargs="*",
        default=["SVM", "SVM-balanced", "RF", "RF-balanced", "GB", "XGB", "EfficientKAN", "FastKAN", "FasterKAN", "ChebyshevKAN"],
        help="Models to run.",
    )
    p.add_argument(
        "--modes",
        nargs="*",
        default=["NoAug", "DiffAug"],
        choices=["NoAug", "DiffAug", "DiffAug-noAlign", "SMOTE", "BorderlineSMOTE", "KMeansSMOTE", "TabDDPM"],
        help=(
            "Training modes for KAN backbones. "
            "Supported augmentation baselines are NoAug, DiffAug, DiffAug-noAlign, SMOTE, "
            "BorderlineSMOTE, KMeansSMOTE, and TabDDPM. Classical baselines still run in NoAug only."
        ),
    )

    p.add_argument(
        "--extra_python_path",
        action="append",
        default=[],
        help="Extra path added to sys.path before importing KAN libraries. Repeatable.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)

    p.add_argument("--no_fs", action="store_true", help="Disable fold-local feature selection.")
    p.add_argument("--no_diffusion", action="store_true", help="Disable diffusion augmentation.")
    p.add_argument("--no_hdbscan", action="store_true", help="Ablation: disable HDBSCAN allocation.")
    p.add_argument("--no_anchor", action="store_true", help="Ablation: disable anchored DDIM sampling.")
    p.add_argument("--no_knn", action="store_true", help="Ablation: disable within-class kNN filtering.")

    p.add_argument("--gamma", type=float, default=1.25, help="Overshoot factor gamma.")
    p.add_argument("--w_syn", type=float, default=0.1, help="Synthetic sample weight.")
    p.add_argument("--keep_q", type=float, default=0.98, help="Retention quantile Keep_Q.")
    p.add_argument("--knn_k", type=int, default=5, help="k in within-class kNN filter.")
    p.add_argument(
        "--min_real_for_synthesis",
        type=int,
        default=2,
        help=(
            "Rare-class guard shared by DiffAug and SMOTE-family baselines. "
            "Any class with fewer than this many real training samples in the current fold "
            "receives quota=0. Use 1 to recover the previous behavior; use 3 for a stricter guard."
        ),
    )
    p.add_argument("--smote_k_neighbors", type=int, default=5,
                   help="Requested k_neighbors for SMOTE-family baselines; reduced automatically per fold if needed.")
    p.add_argument("--smote_m_neighbors", type=int, default=10,
                   help="Requested m_neighbors for BorderlineSMOTE; reduced automatically per fold if needed.")
    p.add_argument("--borderline_kind", type=str, default="borderline-1",
                   choices=["borderline-1", "borderline-2"],
                   help="BorderlineSMOTE variant.")
    p.add_argument("--kmeans_n_clusters", type=int, default=5,
                   help="Starting number of clusters for KMeansSMOTE; the code backs off to fewer clusters on hard folds.")

    p.add_argument("--save_synthesis", action="store_true", help="Save synthetic bundles.")
    p.add_argument("--save_fold_artifacts", action="store_true", help="Save fold-local artifacts.")

    p.add_argument("--pre_screen_topk", type=int, default=None,
                   help="Optional speed-up filter before mRMR. Not part of the paper default.")
    p.add_argument("--mrmr_candidate_pool", type=int, default=4000,
                   help="Candidate pool size used before redundancy filtering; internal implementation detail.")
    p.add_argument("--mrmr_topk", type=int, default=1000)
    p.add_argument("--mrmr_redundancy_threshold", type=float, default=0.9)
    p.add_argument("--boruta_n_estimators", type=int, default=300)
    p.add_argument("--boruta_max_iter", type=int, default=200)
    p.add_argument("--boruta_alpha", type=float, default=0.05,
                   help="Implementation detail not specified explicitly in the manuscript.")
    p.add_argument("--fs_min_keep", type=int, default=10)
    p.add_argument("--fs_fallback_topk", type=int, default=200)

    p.add_argument("--alignment_bins", type=int, default=32,
                   help="Fixed number of bins per feature for JS approximation.")
    p.add_argument("--diff_lambda_js", type=float, default=10.0,
                   help="Weight of JS alignment loss in diffusion training. Ignored by DiffAug-noAlign.")
    p.add_argument("--diff_lambda_mmd", type=float, default=10.0,
                   help="Weight of MMD alignment loss in diffusion training. Ignored by DiffAug-noAlign.")

    p.add_argument("--log_file", default="", help="Optional explicit txt log path.")

    ns = p.parse_args()

    ns.data_path = os.path.normpath(os.path.expanduser(str(ns.data_path)))
    ns.results_path = os.path.normpath(os.path.expanduser(str(ns.results_path)))
    ns.min_real_for_synthesis = max(1, int(ns.min_real_for_synthesis))

    models = [str(m) for m in ns.models]
    modes = [str(m) for m in ns.modes]
    dataset_ids = [str(x) for x in ns.dataset_ids] if ns.dataset_ids else None
    extra_python_path = [str(x) for x in ns.extra_python_path]

    return Args(
        data_path=str(ns.data_path),
        results_path=str(ns.results_path),
        run_name=str(ns.run_name),
        dataset_ids=dataset_ids,
        models=models,
        modes=modes,
        smote_k_neighbors=int(ns.smote_k_neighbors),
        smote_m_neighbors=int(ns.smote_m_neighbors),
        borderline_kind=str(ns.borderline_kind),
        kmeans_n_clusters=int(ns.kmeans_n_clusters),
        extra_python_path=extra_python_path,
        seed=int(ns.seed),
        val_ratio=float(ns.val_ratio),
        batch_size=int(ns.batch_size),
        num_epochs=int(ns.num_epochs),
        patience=int(ns.patience),
        no_fs=bool(ns.no_fs),
        no_diffusion=bool(ns.no_diffusion),
        no_hdbscan=bool(ns.no_hdbscan),
        no_anchor=bool(ns.no_anchor),
        no_knn=bool(ns.no_knn),
        gamma=float(ns.gamma),
        w_syn=float(ns.w_syn),
        keep_q=float(ns.keep_q),
        knn_k=int(ns.knn_k),
        min_real_for_synthesis=int(ns.min_real_for_synthesis),
        save_synthesis=bool(ns.save_synthesis),
        save_fold_artifacts=bool(ns.save_fold_artifacts),
        pre_screen_topk=None if ns.pre_screen_topk is None else int(ns.pre_screen_topk),
        mrmr_candidate_pool=None if ns.mrmr_candidate_pool is None else int(ns.mrmr_candidate_pool),
        mrmr_topk=int(ns.mrmr_topk),
        mrmr_redundancy_threshold=float(ns.mrmr_redundancy_threshold),
        boruta_n_estimators=int(ns.boruta_n_estimators),
        boruta_max_iter=int(ns.boruta_max_iter),
        boruta_alpha=float(ns.boruta_alpha),
        fs_min_keep=int(ns.fs_min_keep),
        fs_fallback_topk=int(ns.fs_fallback_topk),
        alignment_bins=int(ns.alignment_bins),
        diff_lambda_js=float(ns.diff_lambda_js),
        diff_lambda_mmd=float(ns.diff_lambda_mmd),
        log_file=str(ns.log_file or ""),
    )

AUGMENT_MODES = {"DiffAug", "DiffAug-noAlign", "SMOTE", "BorderlineSMOTE", "KMeansSMOTE", "TabDDPM"}

DIFFUSION_MODES = {"DiffAug", "DiffAug-noAlign"}

def is_diffusion_mode(mode: str) -> bool:
    return str(mode) in DIFFUSION_MODES

def resolve_diffusion_lambdas(mode: str, args: Args) -> Tuple[float, float]:
    if str(mode) == "DiffAug-noAlign":
        return 0.0, 0.0
    return float(args.diff_lambda_js), float(args.diff_lambda_mmd)
