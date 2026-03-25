

from __future__ import annotations


import os
import sys
import io
import gzip
import json
import math
import time
import copy
import hashlib
import random
import zipfile
import argparse
import warnings
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models.mlp import (
        PlainMLP,
        count_trainable_parameters,
        solve_ratio_preserving_two_hidden_mlp,
    )
except ImportError:  # pragma: no cover
    from fodikan.models.mlp import (
        PlainMLP,
        count_trainable_parameters,
        solve_ratio_preserving_two_hidden_mlp,
    )

from scipy.stats import binomtest, wilcoxon
from scipy.spatial.distance import pdist

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN, KMeans

warnings.filterwarnings("ignore", message=".*verbose parameter is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False

# =============================================================================
# BACKBONE REGISTRY
# =============================================================================
KAN_MODEL_NAMES = {"EfficientKAN", "FastKAN", "FasterKAN"}
MLP_PARAM_TARGET_MAP = {
    "MLPParam_EfficientKAN": "EfficientKAN",
    "MLPParam_FastKAN": "FastKAN",
    "MLPParam_FasterKAN": "FasterKAN",
}
MLP_MODEL_NAMES = {"MLPShape"} | set(MLP_PARAM_TARGET_MAP.keys())
TORCH_MODEL_NAMES = KAN_MODEL_NAMES | MLP_MODEL_NAMES


# =============================================================================
# GLOBALS / REPRODUCIBILITY
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DATA_DIR = "data"
BASE_RESULTS_PATH = "results"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# LOGGING
# =============================================================================
class TxtLogger:
    def __init__(self, path: str, also_print: bool = True) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.also_print = bool(also_print)
        self._fh = open(path, "w", encoding="utf-8")

    def write(self, text: str = "") -> None:
        self._fh.write(str(text) + "\n")
        self._fh.flush()
        if self.also_print:
            print(text)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _fmt_table(rows: List[List[str]], headers: List[str]) -> str:
    if not rows:
        return " | ".join(headers)
    widths = [len(str(h)) for h in headers]
    for r in rows:
        for j, v in enumerate(r):
            widths[j] = max(widths[j], len(str(v)))
    def fmt_row(r: Sequence[Any]) -> str:
        return "  ".join(str(r[j]).ljust(widths[j]) for j in range(len(headers)))
    out = [fmt_row(headers), "  ".join("-" * w for w in widths)]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


# =============================================================================
# CLI / CONFIG
# =============================================================================
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

    mlp_shape_hidden: List[int]
    mlp_activation: str
    mlp_dropout: float
    mlp_param_match_ratio: float

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
        default=["SVM", "RF", "GB", "XGB", "EfficientKAN", "FastKAN", "FasterKAN", "MLPShape", "MLPParam_EfficientKAN", "MLPParam_FastKAN", "MLPParam_FasterKAN"],
        help="Models to run.",
    )
    p.add_argument(
        "--modes",
        nargs="*",
        default=["NoAug", "DiffAug"],
        choices=["NoAug", "DiffAug", "DiffAug-noAlign", "SMOTE", "BorderlineSMOTE", "KMeansSMOTE"],
        help=(
            "Training modes for KAN backbones. "
            "Supported augmentation baselines are NoAug, DiffAug, DiffAug-noAlign, SMOTE, "
            "BorderlineSMOTE, and KMeansSMOTE. Classical baselines still run in NoAug only."
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

    p.add_argument(
        "--mlp_shape_hidden",
        nargs="*",
        type=int,
        default=[128, 64],
        help="Hidden widths for the shape-matched MLP control.",
    )
    p.add_argument(
        "--mlp_activation",
        choices=["relu", "gelu", "silu"],
        default="silu",
        help="Activation used by the MLP controls.",
    )
    p.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.0,
        help="Dropout used in MLP controls.",
    )
    p.add_argument(
        "--mlp_param_match_ratio",
        type=float,
        default=0.5,
        help="For param-matched MLPs, enforce hidden widths [h1, round(h1 * ratio)].",
    )

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

    # BoMGene / FS defaults aligned to manuscript Table 3
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
        mlp_shape_hidden=[int(x) for x in ns.mlp_shape_hidden],
        mlp_activation=str(ns.mlp_activation),
        mlp_dropout=float(ns.mlp_dropout),
        mlp_param_match_ratio=float(ns.mlp_param_match_ratio),
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


AUGMENT_MODES = {"DiffAug", "DiffAug-noAlign", "SMOTE", "BorderlineSMOTE", "KMeansSMOTE"}
DIFFUSION_MODES = {"DiffAug", "DiffAug-noAlign"}


def is_diffusion_mode(mode: str) -> bool:
    return str(mode) in DIFFUSION_MODES


def resolve_diffusion_lambdas(mode: str, args: Args) -> Tuple[float, float]:
    if str(mode) == "DiffAug-noAlign":
        return 0.0, 0.0
    return float(args.diff_lambda_js), float(args.diff_lambda_mmd)

# =============================================================================
# DATA I/O
# =============================================================================
def df_map(df: pd.DataFrame, func):
    try:
        return df.map(func)
    except AttributeError:
        return df.applymap(func)


def _read_csv_or_excel_robust(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        sig = fh.read(4)

    if sig[:2] == b"PK":
        with open(path, "rb") as fh:
            data = fh.read()
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if any(n.startswith("xl/") for n in names):
                    return pd.read_excel(io.BytesIO(data))
                csv_names = [n for n in names if n.lower().endswith(".csv")]
                txt_names = [n for n in names if n.lower().endswith((".txt", ".tsv"))]
                if csv_names:
                    with zf.open(csv_names[0]) as fh:
                        return pd.read_csv(fh, sep=None, engine="python")
                if txt_names:
                    with zf.open(txt_names[0]) as fh:
                        return pd.read_csv(fh, sep=None, engine="python")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Broken ZIP/XLSX file: {path}") from exc

    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception as exc:
            last_err = exc
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin1", on_bad_lines="skip")
    except Exception as exc:
        raise ValueError(f"Cannot read file: {path}. Last error: {exc} | Previous: {last_err}") from exc


def load_data_from_csv_smart(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = _read_csv_or_excel_robust(path)
    df = df.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError(f"{os.path.basename(path)} must contain >=2 columns (features + label).")

    X_df = df.iloc[:, :-1].copy()
    y_raw = df.iloc[:, -1].values

    def is_suspicious(v: Any) -> bool:
        return isinstance(v, (dict, list)) or (isinstance(v, str) and "OrderedDict" in v)

    susp = df_map(X_df, is_suspicious)
    if bool(susp.values.any()):
        locs = np.argwhere(susp.values)[:10]
        print(f"[Warning] Suspicious cells in {os.path.basename(path)}:")
        for r, c in locs:
            print(f"  row={r}, col={X_df.columns[c]}, value={repr(X_df.iat[r, c])}")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    all_nan_cols = X_df.columns[X_df.isna().all()].tolist()
    if all_nan_cols:
        X_df = X_df.drop(columns=all_nan_cols)

    if X_df.shape[1] == 0:
        raise ValueError(f"All feature columns became NaN after numeric conversion: {path}")

    X_df = X_df.apply(lambda s: s.fillna(s.median()), axis=0)

    X = X_df.to_numpy(dtype=np.float32, copy=False)
    y = LabelEncoder().fit_transform(y_raw).astype(np.int64, copy=False)
    if len(np.unique(y)) < 2:
        raise ValueError(f"{path}: only one class after encoding.")
    feature_names = [str(c) for c in X_df.columns]
    return X, y, feature_names


def load_dataset_dhkan_style(dataset_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    path = os.path.join(dataset_dir, "data.trn.gz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing data.trn.gz at: {path}")
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, header=None, sep=r"\s+")
    if df.shape[1] < 2:
        raise ValueError(f"{path}: expected >=2 columns (features + label)")
    X = df.iloc[:, :-1].values.astype(np.float32, copy=False)
    y_raw = df.iloc[:, -1].values
    y = LabelEncoder().fit_transform(y_raw).astype(np.int64, copy=False)
    feature_names = [f"Gene_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


@dataclass
class InputItem:
    kind: str
    dataset_id: str
    path: str


def dataset_id_from_filename(file_name: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    for suf in ("_selected", "_transformed", "_merged"):
        if stem.endswith(suf):
            return stem[:-len(suf)]
    return stem


def list_input_files(base_dir: str) -> List[str]:
    out: List[str] = []
    for f in sorted(os.listdir(base_dir)):
        p = os.path.join(base_dir, f)
        if not os.path.isfile(p):
            continue
        if f.lower().endswith(("_generated.csv", "_generated.tsv", "_generated.txt")):
            continue
        if f.lower().endswith((".csv", ".tsv", ".txt", ".xlsx")):
            out.append(f)
    return out


def discover_inputs(base_dir: str, dataset_ids: Optional[List[str]] = None) -> List[InputItem]:
    """
    Discover datasets under base_dir.

    Supported preferred layout:
        base_dir/<dataset_id>/data.trn.gz

    If such subdirectories are found, they are used preferentially. Otherwise the
    function falls back to flat CSV/TSV/TXT/XLSX files directly under base_dir.
    """
    items: List[InputItem] = []
    if not os.path.isdir(base_dir):
        return items

    wanted = set(dataset_ids) if dataset_ids else None

    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dhkan_dirs = []
    for d in subdirs:
        if os.path.isfile(os.path.join(base_dir, d, "data.trn.gz")):
            dhkan_dirs.append(d)

    if dhkan_dirs:
        dhkan_dirs = sorted(dhkan_dirs, key=lambda x: int(x) if str(x).isdigit() else str(x))
        for ds_id in dhkan_dirs:
            if wanted is not None and str(ds_id) not in wanted:
                continue
            items.append(InputItem(kind="dhkan", dataset_id=str(ds_id), path=os.path.join(base_dir, ds_id)))
        return items

    for f in list_input_files(base_dir):
        ds_id = dataset_id_from_filename(f)
        if wanted is not None and str(ds_id) not in wanted:
            continue
        items.append(InputItem(kind="csv", dataset_id=str(ds_id), path=os.path.join(base_dir, f)))
    return items


# =============================================================================
# METRICS / CV HELPERS
# =============================================================================
def fixed_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    labels = list(range(num_classes))
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "Precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "Recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "F1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)) if len(y_true) else 0.0,
    }


def macro_f1_defined(y: np.ndarray, num_classes: int) -> bool:
    if y is None or len(y) == 0:
        return False
    counts = np.bincount(np.asarray(y, dtype=int), minlength=num_classes)
    return bool(np.all(counts > 0))


def choose_size_aware_cv(y: np.ndarray, seed: int) -> Tuple[Any, str]:
    """
    Paper-style outer-CV policy:
    - largest feasible StratifiedKFold with k in {10,5,4,3,2}
    - otherwise largest feasible KFold with k in {10,5,4,3,2}
    - otherwise LOOCV
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    counts = np.unique(y, return_counts=True)[1] if n > 0 else np.array([], dtype=int)

    for k in [10, 5, 4, 3, 2]:
        if n >= k and counts.size > 0 and int(counts.min()) >= k:
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed), f"StratifiedKFold(k={k})"

    for k in [10, 5, 4, 3, 2]:
        if n >= k:
            return KFold(n_splits=k, shuffle=True, random_state=seed), f"KFold(k={k})"

    return LeaveOneOut(), "LOOCV"


def safe_train_val_split_indices(
    y_train_outer: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Create real-only (D_tr, D_val) split.
    We adjust the realized validation size to ensure:
    - at least one validation sample per class
    - at least one remaining real training sample per class
    If infeasible, we return D_val = empty and let the caller use inner CV on D_tr only.
    """
    y = np.asarray(y_train_outer, dtype=int)
    n = len(y)
    idx = np.arange(n, dtype=int)

    meta = {
        "feasible": False,
        "realized_val_ratio": 0.0,
        "n_val": 0,
        "val_class_counts": {},
        "train_class_counts": {},
        "reason": "",
    }

    if n < 2 or val_ratio <= 0.0:
        meta["reason"] = "n<2 or val_ratio<=0"
        return idx, np.array([], dtype=int), meta

    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    if n_classes == 0:
        meta["reason"] = "no classes"
        return idx, np.array([], dtype=int), meta
    if np.any(counts < 2):
        meta["reason"] = "at least one class has <2 samples; real-only D_val infeasible"
        return idx, np.array([], dtype=int), meta

    min_val_total = n_classes
    max_val_total = n - n_classes
    if max_val_total < min_val_total:
        meta["reason"] = "cannot preserve >=1 train sample per class"
        return idx, np.array([], dtype=int), meta

    target = int(round(float(val_ratio) * n))
    target = max(target, min_val_total)
    target = min(target, max_val_total)

    # Base: one validation sample per class
    alloc = {int(c): 1 for c in classes.tolist()}
    remaining = int(target - n_classes)

    # Largest-remainder allocation with per-class capacity = count - 2
    extras_cap = {int(c): int(k - 2) for c, k in zip(classes.tolist(), counts.tolist())}
    if remaining > 0:
        desired_extra = (counts.astype(float) * float(val_ratio)) - 1.0
        desired_extra = np.clip(desired_extra, 0.0, None)

        base_extra = np.floor(desired_extra).astype(int)
        base_extra = np.minimum(base_extra, np.array([extras_cap[int(c)] for c in classes], dtype=int))

        used = int(base_extra.sum())
        for c, extra in zip(classes.tolist(), base_extra.tolist()):
            alloc[int(c)] += int(extra)

        remaining -= used

        if remaining > 0:
            remainders = []
            for c, wanted, extra in zip(classes.tolist(), desired_extra.tolist(), base_extra.tolist()):
                cap_left = extras_cap[int(c)] - int(extra)
                if cap_left > 0:
                    remainders.append((wanted - math.floor(wanted), int(c)))
            remainders.sort(reverse=True)
            for _, c in remainders:
                if remaining <= 0:
                    break
                if alloc[int(c)] < int(np.where(classes == c, counts, 0).max()) - 1:
                    alloc[int(c)] += 1
                    remaining -= 1

        if remaining > 0:
            cyc = [int(c) for c in classes.tolist() if alloc[int(c)] < int(np.where(classes == c, counts, 0).max()) - 1]
            while remaining > 0 and cyc:
                next_cyc = []
                for c in cyc:
                    max_for_c = int(np.where(classes == c, counts, 0).max()) - 1
                    if alloc[int(c)] < max_for_c and remaining > 0:
                        alloc[int(c)] += 1
                        remaining -= 1
                    if alloc[int(c)] < max_for_c:
                        next_cyc.append(c)
                cyc = next_cyc

    rng = np.random.RandomState(seed)
    val_idx_parts: List[np.ndarray] = []
    train_idx_parts: List[np.ndarray] = []
    for c in classes.tolist():
        class_idx = idx[y == int(c)]
        rng.shuffle(class_idx)
        n_val_c = int(alloc[int(c)])
        val_idx_parts.append(class_idx[:n_val_c])
        train_idx_parts.append(class_idx[n_val_c:])

    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], dtype=int)
    tr_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.array([], dtype=int)

    # Shuffle combined splits for training stochasticity
    rng.shuffle(tr_idx)
    rng.shuffle(val_idx)

    y_val = y[val_idx]
    y_tr = y[tr_idx]

    meta["feasible"] = True
    meta["n_val"] = int(len(val_idx))
    meta["realized_val_ratio"] = float(len(val_idx)) / float(max(1, n))
    meta["val_class_counts"] = {int(k): int(v) for k, v in Counter(y_val.tolist()).items()}
    meta["train_class_counts"] = {int(k): int(v) for k, v in Counter(y_tr.tolist()).items()}
    meta["reason"] = "ok"
    return tr_idx.astype(int), val_idx.astype(int), meta


# =============================================================================
# LOADERS
# =============================================================================
def make_loader_with_weights(
    X: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    if w is None:
        w = np.ones(len(y), dtype=np.float32)
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(w, dtype=torch.float32),
    )
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        generator=g,
        persistent_workers=False,
    )


def make_eval_loader(X: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False,
    )


# =============================================================================
# BOmGene: (fold-local)
# =============================================================================
def _bh_fdr(pvals: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    if m == 0:
        return np.array([], dtype=int), 0.0
    order = np.argsort(pvals)
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = pvals[order] <= thresh
    if not np.any(passed):
        return np.array([], dtype=int), 0.0
    k = int(np.max(np.where(passed)[0]))
    cutoff = float(thresh[k])
    keep = np.where(pvals <= cutoff)[0]
    return keep.astype(int), cutoff


@dataclass
class BoMGeneStats:
    n_input: int = 0
    n_prescreen: int = 0
    n_mrmr: int = 0
    n_boruta: int = 0
    time_prescreen: float = 0.0
    time_mrmr: float = 0.0
    time_boruta: float = 0.0
    fdr_cutoff: float = 0.0
    from_cache: bool = False


class BoMGeneSelector:
    def __init__(
        self,
        seed: int,
        cache_dir: Optional[str],
        pre_screen_topk: Optional[int],
        mrmr_candidate_pool: Optional[int],
        mrmr_topk: int,
        redundancy_threshold: float,
        boruta_n_estimators: int,
        boruta_max_iter: int,
        boruta_alpha: float,
        min_keep: int,
        fallback_topk: int,
    ) -> None:
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.pre_screen_topk = None if pre_screen_topk is None else int(pre_screen_topk)
        self.mrmr_candidate_pool = None if mrmr_candidate_pool is None else int(mrmr_candidate_pool)
        self.mrmr_topk = int(mrmr_topk)
        self.redundancy_threshold = float(redundancy_threshold)
        self.boruta_n_estimators = int(boruta_n_estimators)
        self.boruta_max_iter = int(boruta_max_iter)
        self.boruta_alpha = float(boruta_alpha)
        self.min_keep = int(min_keep)
        self.fallback_topk = int(fallback_topk)

        self.selected_indices_: Optional[np.ndarray] = None
        self.selected_names_: Optional[List[str]] = None
        self.stats_: BoMGeneStats = BoMGeneStats()

    def _cache_path(self, dataset_id: str, fold_idx: int, train_hash: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"fs_{dataset_id}_fold{fold_idx:03d}_{train_hash}.json")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        dataset_id: str,
        fold_idx: int,
        train_hash: str,
    ) -> "BoMGeneSelector":
        n, p = X.shape
        self.stats_ = BoMGeneStats(n_input=int(p))
        cache_path = self._cache_path(dataset_id, fold_idx, train_hash)

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                sel = np.array(cached["selected_indices"], dtype=int)
                if sel.size > 0 and int(sel.max()) < p:
                    self.selected_indices_ = sel
                    self.selected_names_ = [feature_names[i] for i in sel]
                    self.stats_ = BoMGeneStats(**cached.get("stats", {}))
                    self.stats_.from_cache = True
                    return self
            except Exception:
                pass

        rng = np.random.RandomState(self.seed + fold_idx)

        # Optional implementation-level prescreen (default: None for paper alignment)
        prescreen_idx = np.arange(p, dtype=int)
        if self.pre_screen_topk is not None and p > int(self.pre_screen_topk):
            t0 = time.time()
            f_vals, _ = f_classif(X, y)
            f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
            top = np.argsort(f_vals)[-int(self.pre_screen_topk):]
            prescreen_idx = np.sort(top.astype(int))
            self.stats_.time_prescreen = time.time() - t0
        self.stats_.n_prescreen = int(len(prescreen_idx))

        X_ps = X[:, prescreen_idx]

        # mRMR: MI relevance + redundancy threshold
        t0 = time.time()
        mi = mutual_info_classif(X_ps, y, random_state=int(rng.randint(0, 10 ** 9)))
        mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        rank = np.argsort(mi)[::-1]

        if self.mrmr_candidate_pool is not None:
            cand = rank[: min(len(rank), max(int(self.mrmr_candidate_pool), self.mrmr_topk))]
        else:
            cand = rank

        X_cand = X_ps[:, cand].astype(np.float32, copy=False)
        X_cand = X_cand - X_cand.mean(axis=0, keepdims=True)
        X_cand = X_cand / (X_cand.std(axis=0, keepdims=True) + 1e-12)

        corr = (X_cand.T @ X_cand) / float(max(1, n - 1))
        corr = np.abs(corr)

        selected_local: List[int] = []
        for i in range(len(cand)):
            if not selected_local:
                selected_local.append(i)
            else:
                if float(np.max(corr[i, selected_local])) < self.redundancy_threshold:
                    selected_local.append(i)
            if len(selected_local) >= self.mrmr_topk:
                break

        if len(selected_local) < self.min_keep:
            selected_local = list(range(min(len(cand), self.mrmr_topk)))

        mrmr_idx_ps = cand[np.array(selected_local, dtype=int)]
        mrmr_idx_orig = prescreen_idx[mrmr_idx_ps]
        self.stats_.n_mrmr = int(len(mrmr_idx_orig))
        self.stats_.time_mrmr = time.time() - t0

        # Boruta (RF importance vs shadow features)
        t0 = time.time()
        X_mrmr = X[:, mrmr_idx_orig]
        d = X_mrmr.shape[1]

        if d <= self.min_keep:
            final_idx = mrmr_idx_orig
            fdr_cutoff = 0.0
        else:
            hits = np.zeros(d, dtype=int)
            rng_b = np.random.RandomState(self.seed + 1000 + fold_idx)

            for _ in range(self.boruta_max_iter):
                X_shadow = X_mrmr.copy()
                for j in range(d):
                    rng_b.shuffle(X_shadow[:, j])
                X_b = np.hstack([X_mrmr, X_shadow])

                rf = RandomForestClassifier(
                    n_estimators=self.boruta_n_estimators,
                    n_jobs=-1,
                    random_state=int(rng_b.randint(0, 10 ** 9)),
                )
                rf.fit(X_b, y)
                imp = rf.feature_importances_
                imp_real = imp[:d]
                imp_shadow = imp[d:]
                thr = float(np.max(imp_shadow))
                hits += (imp_real > thr).astype(int)

            pvals = np.array([
                binomtest(int(hits[j]), int(self.boruta_max_iter), p=0.5, alternative="greater").pvalue
                for j in range(d)
            ], dtype=float)

            keep_local, fdr_cutoff = _bh_fdr(pvals, self.boruta_alpha)
            if keep_local.size < self.min_keep:
                topk = int(min(max(self.fallback_topk, self.min_keep), d))
                keep_local = np.argsort(hits)[-topk:]
                keep_local = np.sort(keep_local.astype(int))

            final_idx = mrmr_idx_orig[keep_local]

        self.stats_.time_boruta = time.time() - t0
        self.stats_.n_boruta = int(len(final_idx))
        self.stats_.fdr_cutoff = float(fdr_cutoff)

        if len(final_idx) < self.min_keep:
            k = int(min(max(self.min_keep, self.fallback_topk), len(mrmr_idx_orig)))
            final_idx = mrmr_idx_orig[:k]

        self.selected_indices_ = np.array(final_idx, dtype=int)
        self.selected_names_ = [feature_names[i] for i in self.selected_indices_]

        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "selected_indices": self.selected_indices_.tolist(),
                            "selected_names": self.selected_names_,
                            "stats": asdict(self.stats_),
                        },
                        fh,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_indices_ is None or len(self.selected_indices_) == 0:
            return X
        return X[:, self.selected_indices_]

    def get_kept_names(self) -> List[str]:
        return list(self.selected_names_ or [])


# =============================================================================
# DIFFUSION MODEL / ALIGNMENT LOSS
# =============================================================================
@dataclass
class DiffusionConfig:
    T: int
    hidden: int = 64
    blocks: int = 3
    dropout: float = 0.5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 400
    patience: int = 50
    batch_size: int = 16
    lambda_js: float = 10.0
    lambda_mmd: float = 10.0
    lambda_recon: float = 1.0
    self_condition: bool = True
    ddim_steps: int = 50
    ddim_eta: float = 0.0
    start_t_ratio_steps: float = 0.5
    seed: int = 2025
    alignment_bins: int = 32


def pick_diffusion_config(
    n_samples: int,
    alignment_bins: int,
    seed: int,
    lambda_js: float = 10.0,
    lambda_mmd: float = 10.0,
) -> DiffusionConfig:
    if n_samples >= 200:
        return DiffusionConfig(
            T=500,
            ddim_steps=50,
            start_t_ratio_steps=0.5,
            epochs=250,
            patience=30,
            alignment_bins=alignment_bins,
            seed=seed,
            lambda_js=float(lambda_js),
            lambda_mmd=float(lambda_mmd),
        )
    if n_samples >= 100:
        return DiffusionConfig(
            T=300,
            ddim_steps=30,
            start_t_ratio_steps=0.4,
            epochs=300,
            patience=40,
            alignment_bins=alignment_bins,
            seed=seed,
            lambda_js=float(lambda_js),
            lambda_mmd=float(lambda_mmd),
        )
    return DiffusionConfig(
        T=200,
        ddim_steps=20,
        start_t_ratio_steps=0.25,
        epochs=400,
        patience=50,
        alignment_bins=alignment_bins,
        seed=seed,
        lambda_js=float(lambda_js),
        lambda_mmd=float(lambda_mmd),
    )


class ResidualBlock(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(d, d)
        self.act = torch.nn.SiLU()
        self.norm = torch.nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.act(self.fc(x)))


class DiffusionDenoiser(torch.nn.Module):
    def __init__(self, d_in: int, n_classes: int, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.t_emb = torch.nn.Embedding(cfg.T, 32)
        self.y_emb = torch.nn.Embedding(n_classes, 16)
        dim = d_in + 32 + 16 + (d_in if cfg.self_condition else 0)
        self.fc_in = torch.nn.Linear(dim, cfg.hidden)
        self.blocks = torch.nn.Sequential(*[ResidualBlock(cfg.hidden) for _ in range(cfg.blocks)])
        self.fc_out = torch.nn.Linear(cfg.hidden, d_in)
        self.drop = torch.nn.Dropout(cfg.dropout)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor, x0_sc: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [x_t, self.t_emb(t), self.y_emb(y)]
        if self.cfg.self_condition:
            parts.append(x0_sc if x0_sc is not None else torch.zeros_like(x_t))
        h = torch.cat(parts, dim=1)
        h = self.drop(torch.nn.functional.silu(self.fc_in(h)))
        h = self.blocks(h)
        h = self.drop(h)
        return self.fc_out(h)


def get_beta_schedule(T: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0, T, T + 1, device=DEVICE)
    a_cum = torch.cos((x / T + 0.008) / (1 + 0.008) * np.pi / 2) ** 2
    a_cum = a_cum / a_cum[0]
    betas = 1 - (a_cum[1:] / a_cum[:-1])
    betas = torch.clamp(betas, 1e-8, 0.999)
    return betas, a_cum


def predict_x0_from_xt(x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor, a_cum: torch.Tensor) -> torch.Tensor:
    alpha_bar = a_cum[t].view(-1, 1)
    return (x_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar + 1e-8)


@dataclass
class HistogramContext:
    centers: torch.Tensor   # [d, bins]
    width: torch.Tensor     # [d]


def build_histogram_context(X_real: np.ndarray, bins: int) -> HistogramContext:
    X = np.asarray(X_real, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X_real must be 2D")
    minv = X.min(axis=0)
    maxv = X.max(axis=0)
    span = maxv - minv
    pad = 0.05 * span + 1e-3
    minv = minv - pad
    maxv = maxv + pad
    same = span < 1e-6
    minv[same] = minv[same] - 0.5
    maxv[same] = maxv[same] + 0.5

    t = torch.linspace(0.0, 1.0, bins, device=DEVICE).view(1, bins)
    min_t = torch.tensor(minv, dtype=torch.float32, device=DEVICE).view(-1, 1)
    max_t = torch.tensor(maxv, dtype=torch.float32, device=DEVICE).view(-1, 1)
    centers = min_t + (max_t - min_t) * t
    width = ((max_t - min_t).squeeze(1) / max(1, bins - 1)).clamp(min=1e-3)
    return HistogramContext(centers=centers, width=width)


def fixed_soft_hist(x: torch.Tensor, ctx: HistogramContext) -> torch.Tensor:
    # x: [n, d], output: [d, bins]
    diff = (x.unsqueeze(-1) - ctx.centers.unsqueeze(0)) / ctx.width.view(1, -1, 1)
    h = torch.exp(-0.5 * diff.pow(2)).sum(dim=0)
    h = h / (h.sum(dim=1, keepdim=True) + 1e-8)
    return h


def js_divergence_fixed_hist(x_real: torch.Tensor, x_fake: torch.Tensor, ctx: HistogramContext) -> torch.Tensor:
    p = fixed_soft_hist(x_real, ctx)
    q = fixed_soft_hist(x_fake, ctx)
    m = 0.5 * (p + q)
    kl_p = (p * (torch.log(p + 1e-8) - torch.log(m + 1e-8))).sum(dim=1)
    kl_q = (q * (torch.log(q + 1e-8) - torch.log(m + 1e-8))).sum(dim=1)
    return 0.5 * (kl_p + kl_q).mean()


def median_heuristic_sigma(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return 1.0
    d = pdist(X, metric="euclidean")
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(max(np.median(d), 1e-3))


def build_sigma_by_class(X_real: np.ndarray, y_real: np.ndarray) -> Dict[int, float]:
    out: Dict[int, float] = {}
    y = np.asarray(y_real, dtype=int)
    for c in sorted(np.unique(y).tolist()):
        out[int(c)] = median_heuristic_sigma(X_real[y == int(c)])
    return out


def mmd_rbf_torch(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(max(sigma, 1e-3))
    xx = torch.cdist(x, x, p=2.0).pow(2)
    yy = torch.cdist(y, y, p=2.0).pow(2)
    xy = torch.cdist(x, y, p=2.0).pow(2)
    denom = 2.0 * (sigma ** 2)
    k_xx = torch.exp(-xx / denom).mean()
    k_yy = torch.exp(-yy / denom).mean()
    k_xy = torch.exp(-xy / denom).mean()
    return k_xx + k_yy - 2.0 * k_xy


def js_loss_classwise(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor,
    full_hist_ctx: HistogramContext,
) -> torch.Tensor:
    vals = []
    for c in torch.unique(y):
        mask = (y == c)
        if int(mask.sum().item()) >= 2:
            vals.append(js_divergence_fixed_hist(x_real[mask], x_hat[mask], full_hist_ctx))
    if not vals:
        return torch.tensor(0.0, device=x_real.device)
    return torch.stack(vals).mean()


def mmd_loss_classwise(
    x_real: torch.Tensor,
    x_hat: torch.Tensor,
    y: torch.Tensor,
    sigma_by_class: Dict[int, float],
) -> torch.Tensor:
    vals = []
    for c in torch.unique(y):
        c_int = int(c.item())
        mask = (y == c)
        if int(mask.sum().item()) >= 2:
            vals.append(mmd_rbf_torch(x_real[mask], x_hat[mask], sigma_by_class.get(c_int, 1.0)))
    if not vals:
        return torch.tensor(0.0, device=x_real.device)
    return torch.stack(vals).mean()


def diffusion_training_loss(
    model: DiffusionDenoiser,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    a_cum: torch.Tensor,
    cfg: DiffusionConfig,
    hist_ctx: HistogramContext,
    sigma_by_class: Dict[int, float],
) -> torch.Tensor:
    eps = torch.randn_like(x)
    alpha_bar = a_cum[t].view(-1, 1)
    x_t = torch.sqrt(alpha_bar) * x + torch.sqrt(1.0 - alpha_bar) * eps

    x0_sc = None
    if cfg.self_condition and float(torch.rand(1, device=x.device).item()) < 0.5:
        with torch.no_grad():
            eps_sc = model(x_t, t, y, torch.zeros_like(x))
            x0_sc = predict_x0_from_xt(x_t, t, eps_sc, a_cum).detach()

    eps_pred = model(x_t, t, y, x0_sc)
    x0_hat = predict_x0_from_xt(x_t, t, eps_pred, a_cum)

    recon = cfg.lambda_recon * torch.nn.functional.mse_loss(eps_pred, eps)
    js = cfg.lambda_js * js_loss_classwise(x, x0_hat, y, hist_ctx)
    mmd = cfg.lambda_mmd * mmd_loss_classwise(x, x0_hat, y, sigma_by_class)
    return recon + js + mmd


def train_diffusion_on_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    cfg: DiffusionConfig,
) -> Tuple[DiffusionDenoiser, torch.Tensor]:
    set_seed(cfg.seed)

    X_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tensor = torch.tensor(y_tr, dtype=torch.long)

    n_classes = int(np.max(y_tr)) + 1
    model = DiffusionDenoiser(X_tensor.shape[1], n_classes, cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    _, a_cum = get_beta_schedule(cfg.T)

    hist_ctx = build_histogram_context(X_tr, bins=cfg.alignment_bins)
    sigma_by_class = build_sigma_by_class(X_tr, y_tr)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False,
    )

    best_loss = float("inf")
    best_state = None
    wait = 0

    for _epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            t = torch.randint(0, cfg.T, (xb.shape[0],), device=DEVICE).long()
            loss = diffusion_training_loss(model, xb, yb, t, a_cum, cfg, hist_ctx, sigma_by_class)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            count += 1
        avg = total / max(1, count)
        if avg < best_loss - 1e-6:
            best_loss = avg
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    return model, a_cum


# =============================================================================
# HDBSCAN / DDIM SAMPLING / AUGMENTATION
# =============================================================================
def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / nrm


def cluster_hdbscan_cosine_consistent(X_class: np.ndarray, min_cluster_size: int = 5, min_samples: int = 1) -> np.ndarray:
    n = X_class.shape[0]
    if n <= 1:
        return np.full(n, -1, dtype=int)
    Xn = l2_normalize_rows(X_class.astype(np.float64, copy=False))
    mcs = int(min(max(2, min_cluster_size), n))
    ms = int(min(max(1, min_samples), max(1, n - 1)))
    clusterer = SklearnHDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        n_jobs=-1,
    )
    return clusterer.fit_predict(Xn).astype(int)


@torch.no_grad()
def ddim_step(
    model: DiffusionDenoiser,
    x_t: torch.Tensor,
    y: torch.Tensor,
    t: int,
    t_prev: int,
    a_cum: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    n = x_t.shape[0]
    tt = torch.full((n,), int(t), dtype=torch.long, device=x_t.device)
    at = a_cum[tt].view(-1, 1)
    eps = model(x_t, tt, y, None)
    x0_hat = (x_t - torch.sqrt(1.0 - at) * eps) / torch.sqrt(at + 1e-8)

    if t_prev < 0:
        a_prev = torch.ones_like(at)
    else:
        tt_prev = torch.full((n,), int(t_prev), dtype=torch.long, device=x_t.device)
        a_prev = a_cum[tt_prev].view(-1, 1)

    sigma = eta * torch.sqrt((1 - a_prev) / (1 - at + 1e-8)) * torch.sqrt(torch.clamp(1 - at / (a_prev + 1e-8), min=0.0))
    c = torch.sqrt(torch.clamp(1.0 - a_prev - sigma ** 2, min=0.0))
    noise = torch.randn_like(x_t) if (t_prev >= 0 and eta > 0.0) else torch.zeros_like(x_t)
    return torch.sqrt(a_prev) * x0_hat + c * eps + sigma * noise


@torch.no_grad()
def ddim_sample_anchored(
    model: DiffusionDenoiser,
    X_anchor: torch.Tensor,
    y_anchor: torch.Tensor,
    a_cum: torch.Tensor,
    T: int,
    steps: int,
    eta: float,
    start_t_ratio_steps: float,
) -> torch.Tensor:
    model.eval()
    n = X_anchor.shape[0]
    ddim_ts = np.linspace(T - 1, 0, steps, dtype=int)
    i0 = int((1.0 - float(start_t_ratio_steps)) * (steps - 1))
    i0 = int(np.clip(i0, 0, steps - 1))
    t_start = int(ddim_ts[i0])

    tt0 = torch.full((n,), t_start, dtype=torch.long, device=X_anchor.device)
    a_start = a_cum[tt0].view(-1, 1)
    noise = torch.randn_like(X_anchor)
    x_t = torch.sqrt(a_start) * X_anchor + torch.sqrt(1.0 - a_start) * noise

    for i in range(i0, len(ddim_ts)):
        t = int(ddim_ts[i])
        t_prev = int(ddim_ts[i + 1]) if (i + 1) < len(ddim_ts) else -1
        x_t = ddim_step(model, x_t, y_anchor, t, t_prev, a_cum, eta)
    return x_t


@torch.no_grad()
def ddim_sample_from_noise(
    model: DiffusionDenoiser,
    n_samples: int,
    dim: int,
    y_cond: torch.Tensor,
    a_cum: torch.Tensor,
    T: int,
    steps: int,
    eta: float,
) -> torch.Tensor:
    model.eval()
    ddim_ts = np.linspace(T - 1, 0, steps, dtype=int)
    x_t = torch.randn((n_samples, dim), device=DEVICE)
    for i in range(len(ddim_ts)):
        t = int(ddim_ts[i])
        t_prev = int(ddim_ts[i + 1]) if (i + 1) < len(ddim_ts) else -1
        x_t = ddim_step(model, x_t, y_cond, t, t_prev, a_cum, eta)
    return x_t


def compute_minority_only_quota(
    y_tr: np.ndarray,
    gamma: float,
    min_real_for_synthesis: int = 1,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    y = np.asarray(y_tr, dtype=int)
    thr = int(max(1, min_real_for_synthesis))

    if y.size == 0:
        return {}, {
            "class_counts": {},
            "n_max": 0,
            "target": 0,
            "min_real_for_synthesis": thr,
            "guarded_classes": [],
            "guarded_class_counts": {},
        }

    n_classes = int(np.max(y)) + 1
    cnt = np.bincount(y, minlength=n_classes)
    n_max = int(cnt.max()) if cnt.size else 0
    target = int(math.ceil(float(gamma) * n_max))

    out: Dict[int, int] = {}
    guarded_classes: List[int] = []
    guarded_class_counts: Dict[int, int] = {}

    for c in range(n_classes):
        n_real = int(cnt[c])
        if n_real <= 0:
            out[c] = 0
            continue
        if n_real < thr:
            out[c] = 0
            guarded_classes.append(int(c))
            guarded_class_counts[int(c)] = int(n_real)
            continue
        if n_real == n_max:
            out[c] = 0
        else:
            out[c] = max(0, target - int(n_real))

    meta = {
        "class_counts": {int(i): int(v) for i, v in enumerate(cnt.tolist()) if int(v) > 0},
        "n_max": int(n_max),
        "target": int(target),
        "min_real_for_synthesis": int(thr),
        "guarded_classes": [int(c) for c in guarded_classes],
        "guarded_class_counts": {int(k): int(v) for k, v in guarded_class_counts.items()},
    }
    return out, meta


@dataclass
class AugmentationResult:
    X_aug: np.ndarray
    y_aug: np.ndarray
    w_aug: np.ndarray
    X_syn: Optional[np.ndarray]
    y_syn: Optional[np.ndarray]
    meta: Dict[str, Any]


def generate_candidates_and_filter(
    model: DiffusionDenoiser,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    quota: Dict[int, int],
    a_cum: torch.Tensor,
    cfg: DiffusionConfig,
    use_hdbscan: bool,
    use_anchor: bool,
    filter_knn: bool,
    knn_k: int,
    keep_q: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    X_np = np.asarray(X_tr, dtype=np.float32)
    y_np = np.asarray(y_tr, dtype=int)
    uniq = np.unique(y_np)

    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    def sample_for_indices(idxs: np.ndarray, label: int, n_target: int) -> Optional[torch.Tensor]:
        if n_target <= 0:
            return None
        chunks: List[torch.Tensor] = []
        dim = int(X_np.shape[1])
        remaining = int(n_target)
        while remaining > 0:
            m = min(1024, remaining)
            y_cond = torch.full((m,), int(label), dtype=torch.long, device=DEVICE)
            if use_anchor:
                anchor_choices = np.random.choice(idxs, size=m, replace=(m > len(idxs)))
                X_anchor = torch.tensor(X_np[anchor_choices], dtype=torch.float32, device=DEVICE)
                x_gen = ddim_sample_anchored(
                    model=model,
                    X_anchor=X_anchor,
                    y_anchor=y_cond,
                    a_cum=a_cum,
                    T=cfg.T,
                    steps=cfg.ddim_steps,
                    eta=cfg.ddim_eta,
                    start_t_ratio_steps=cfg.start_t_ratio_steps,
                )
            else:
                x_gen = ddim_sample_from_noise(
                    model=model,
                    n_samples=m,
                    dim=dim,
                    y_cond=y_cond,
                    a_cum=a_cum,
                    T=cfg.T,
                    steps=cfg.ddim_steps,
                    eta=cfg.ddim_eta,
                )
            chunks.append(x_gen.detach().cpu())
            remaining -= m
        return torch.cat(chunks, dim=0) if chunks else None

    for c in uniq.tolist():
        class_idx = np.where(y_np == int(c))[0]
        n_target = int(quota.get(int(c), 0))
        if len(class_idx) == 0 or n_target <= 0:
            continue

        if not use_hdbscan or len(class_idx) < 2:
            x_gen = sample_for_indices(class_idx, int(c), n_target)
            if x_gen is not None and x_gen.shape[0] > 0:
                zs.append(x_gen)
                ys.append(torch.full((x_gen.shape[0],), int(c), dtype=torch.long))
            continue

        labels = cluster_hdbscan_cosine_consistent(X_np[class_idx], min_cluster_size=5, min_samples=1)
        cluster_ids = [cid for cid in np.unique(labels).tolist() if int(cid) != -1]

        if len(cluster_ids) == 0:
            x_gen = sample_for_indices(class_idx, int(c), n_target)
            if x_gen is not None and x_gen.shape[0] > 0:
                zs.append(x_gen)
                ys.append(torch.full((x_gen.shape[0],), int(c), dtype=torch.long))
            continue

        counts = {int(cid): int(np.sum(labels == int(cid))) for cid in cluster_ids}
        total = int(sum(counts.values()))
        alloc = {int(cid): int(math.floor(n_target * counts[int(cid)] / max(1, total))) for cid in cluster_ids}
        diff = int(n_target - sum(alloc.values()))
        if diff > 0:
            for cid in sorted(cluster_ids, key=lambda z: counts[int(z)], reverse=True)[:diff]:
                alloc[int(cid)] += 1

        for cid in cluster_ids:
            n_c = int(alloc[int(cid)])
            if n_c <= 0:
                continue
            sub_idx = class_idx[labels == int(cid)]
            x_gen = sample_for_indices(sub_idx, int(c), n_c)
            if x_gen is not None and x_gen.shape[0] > 0:
                zs.append(x_gen)
                ys.append(torch.full((x_gen.shape[0],), int(c), dtype=torch.long))

    if not zs:
        return None, None, {
            "pre_filter_counts": {},
            "post_filter_counts": {},
            "n_syn_pre_total": 0,
            "n_syn_post_total": 0,
        }

    X_syn_t = torch.cat(zs, dim=0)
    y_syn_t = torch.cat(ys, dim=0)
    pre_counts = Counter(y_syn_t.numpy().astype(int).tolist())

    if filter_knn:
        keep_parts_X: List[torch.Tensor] = []
        keep_parts_y: List[torch.Tensor] = []
        y_syn_np = y_syn_t.numpy().astype(int)
        X_real_norm = l2_normalize_rows(X_np)
        X_syn_np = X_syn_t.numpy().astype(np.float32, copy=False)

        for c in uniq.tolist():
            real_mask = (y_np == int(c))
            syn_mask = (y_syn_np == int(c))
            if int(syn_mask.sum()) == 0:
                continue
            if int(real_mask.sum()) < 2:
                keep_parts_X.append(X_syn_t[syn_mask])
                keep_parts_y.append(y_syn_t[syn_mask])
                continue

            k_use = int(max(2, min(int(knn_k), int(real_mask.sum()) - 1)))
            nbrs = NearestNeighbors(n_neighbors=k_use, metric="euclidean", algorithm="brute")
            nbrs.fit(X_real_norm[real_mask])

            Xs_norm = l2_normalize_rows(X_syn_np[syn_mask])
            dists, _ = nbrs.kneighbors(Xs_norm)
            score = dists.mean(axis=1)
            thr = float(np.quantile(score, float(keep_q)))
            keep = score <= thr
            if np.any(keep):
                keep_parts_X.append(X_syn_t[syn_mask][keep])
                keep_parts_y.append(y_syn_t[syn_mask][keep])

        if keep_parts_X:
            X_syn_t = torch.cat(keep_parts_X, dim=0)
            y_syn_t = torch.cat(keep_parts_y, dim=0)
        else:
            X_syn_t = torch.empty((0, X_syn_t.shape[1]), dtype=X_syn_t.dtype)
            y_syn_t = torch.empty((0,), dtype=y_syn_t.dtype)

    post_counts = Counter(y_syn_t.numpy().astype(int).tolist()) if len(y_syn_t) else Counter()
    X_syn = X_syn_t.numpy().astype(np.float32, copy=False) if len(y_syn_t) else None
    y_syn = y_syn_t.numpy().astype(np.int64, copy=False) if len(y_syn_t) else None

    stats = {
        "pre_filter_counts": {int(k): int(v) for k, v in pre_counts.items()},
        "post_filter_counts": {int(k): int(v) for k, v in post_counts.items()},
        "n_syn_pre_total": int(sum(pre_counts.values())),
        "n_syn_post_total": int(sum(post_counts.values())),
    }
    return X_syn, y_syn, stats



def _quota_to_sampling_strategy(y_tr: np.ndarray, quota: Dict[int, int]) -> Dict[int, int]:
    counts = Counter(np.asarray(y_tr, dtype=int).tolist())
    strategy: Dict[int, int] = {}
    for c, n_real in counts.items():
        c = int(c)
        target = int(n_real) + int(quota.get(c, 0))
        if target > int(n_real):
            strategy[c] = int(target)
    return strategy


def _starting_kmeans_cluster_count(n_samples: int, requested: int) -> int:
    if n_samples <= 1:
        return 1
    return int(max(1, min(int(requested), int(n_samples))))


def augment_training_set_smote_family(
    mode: str,
    dataset_id: str,
    fold_idx: int,
    X_tr_fs: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    args: Args,
    synth_out_dir: Optional[str],
    save_synth: bool,
) -> AugmentationResult:
    if mode not in {"SMOTE", "BorderlineSMOTE", "KMeansSMOTE"}:
        raise ValueError(f"Unsupported SMOTE-family mode: {mode}")

    if not HAS_IMBLEARN:
        raise ImportError("imbalanced-learn is not installed. Install imbalanced-learn to use SMOTE-family baselines.")

    X_real = np.asarray(X_tr_fs, dtype=np.float32)
    y_real = np.asarray(y_tr, dtype=np.int64)

    real_counts = {int(k): int(v) for k, v in Counter(y_real.tolist()).items()}
    quota, quota_meta = compute_minority_only_quota(
        y_real,
        gamma=args.gamma,
        min_real_for_synthesis=args.min_real_for_synthesis,
    )
    strategy = _quota_to_sampling_strategy(y_real, quota)
    guard_counts = {int(k): int(v) for k, v in quota_meta.get("guarded_class_counts", {}).items()}

    meta: Dict[str, Any] = {
        "use_synthesis": True,
        "augmenter": mode,
        "use_hdbscan": False,
        "use_anchor": False,
        "filter_knn": False,
        "knn_k": np.nan,
        "keep_q": np.nan,
        "gamma": float(args.gamma),
        "w_syn": float(args.w_syn),
        "min_real_for_synthesis": int(args.min_real_for_synthesis),
        "lambda_js": float(lambda_js_eff),
        "lambda_mmd": float(lambda_mmd_eff),
        "quota_target": int(quota_meta.get("target", 0)),
        "real_counts": real_counts,
        "quota": {int(k): int(v) for k, v in quota.items()},
        "sampling_strategy": {int(k): int(v) for k, v in strategy.items()},
        "guarded_classes": [int(x) for x in quota_meta.get("guarded_classes", [])],
        "guarded_class_counts": guard_counts,
        "syn_pre_counts": {},
        "syn_post_counts": {},
        "n_syn_pre_total": 0,
        "n_syn_post_total": 0,
        "n_syn": 0,
        "prefix_preserved": False,
        "note": "",
    }

    if len(strategy) == 0:
        meta["note"] = "No positive quota after minority-only allocation."
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    eligible_counts = [real_counts[int(c)] for c in strategy.keys()]
    min_eligible = int(min(eligible_counts)) if eligible_counts else 0
    if min_eligible < 2:
        meta["note"] = "At least one target class has <2 real samples; SMOTE-family generation skipped."
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    k_neighbors = int(max(1, min(int(args.smote_k_neighbors), min_eligible - 1)))
    m_neighbors = int(max(1, min(int(args.smote_m_neighbors), len(y_real) - 1)))
    fold_seed = int(args.seed + 20000 + int(fold_idx))

    X_res = None
    y_res = None
    sampler_params: Dict[str, Any] = {"k_neighbors": int(k_neighbors)}

    try:
        if mode == "SMOTE":
            sampler = SMOTE(
                sampling_strategy=strategy,
                random_state=fold_seed,
                k_neighbors=k_neighbors,
            )
            X_res, y_res = sampler.fit_resample(X_real, y_real)

        elif mode == "BorderlineSMOTE":
            sampler = BorderlineSMOTE(
                sampling_strategy=strategy,
                random_state=fold_seed,
                k_neighbors=k_neighbors,
                m_neighbors=m_neighbors,
                kind=str(args.borderline_kind),
            )
            sampler_params.update({
                "m_neighbors": int(m_neighbors),
                "kind": str(args.borderline_kind),
            })
            X_res, y_res = sampler.fit_resample(X_real, y_real)

        elif mode == "KMeansSMOTE":
            start_clusters = _starting_kmeans_cluster_count(len(y_real), args.kmeans_n_clusters)
            last_exc = None
            tried_clusters: List[int] = []
            for n_clusters in range(start_clusters, 0, -1):
                tried_clusters.append(int(n_clusters))
                try:
                    km = KMeans(n_clusters=int(n_clusters), random_state=fold_seed, n_init=10)
                    sampler = KMeansSMOTE(
                        sampling_strategy=strategy,
                        random_state=fold_seed,
                        k_neighbors=k_neighbors,
                        kmeans_estimator=km,
                        cluster_balance_threshold="auto",
                    )
                    X_res, y_res = sampler.fit_resample(X_real, y_real)
                    sampler_params.update({
                        "kmeans_n_clusters": int(n_clusters),
                        "cluster_balance_threshold": "auto",
                    })
                    break
                except Exception as exc:
                    last_exc = exc
                    X_res, y_res = None, None
            sampler_params["tried_kmeans_clusters"] = tried_clusters
            if X_res is None or y_res is None:
                raise last_exc if last_exc is not None else RuntimeError("KMeansSMOTE failed without explicit exception.")
    except Exception as exc:
        meta["note"] = f"{mode} failed on this fold: {exc}"
        meta["sampler_params"] = sampler_params
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    X_res = np.asarray(X_res, dtype=np.float32)
    y_res = np.asarray(y_res, dtype=np.int64)

    n_real = int(len(y_real))
    n_syn = int(max(0, len(y_res) - n_real))
    prefix_ok = (
        len(y_res) >= n_real and
        np.array_equal(y_res[:n_real], y_real) and
        np.allclose(X_res[:n_real], X_real, atol=1e-7, rtol=1e-6)
    )

    meta["prefix_preserved"] = bool(prefix_ok)
    meta["sampler_params"] = sampler_params

    if n_syn <= 0:
        meta["note"] = f"{mode} returned no synthetic samples."
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    if not prefix_ok:
        meta["note"] = (
            f"{mode} did not preserve original-prefix order under the current imbalanced-learn version; "
            "falling back to unit weights on the whole resampled set and skipping synthetic-only diagnostics."
        )
        return AugmentationResult(
            X_aug=X_res,
            y_aug=y_res,
            w_aug=np.ones(len(y_res), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    X_syn = X_res[n_real:]
    y_syn = y_res[n_real:]
    syn_counts = Counter(y_syn.tolist())

    meta["syn_pre_counts"] = {int(k): int(v) for k, v in syn_counts.items()}
    meta["syn_post_counts"] = {int(k): int(v) for k, v in syn_counts.items()}
    meta["n_syn_pre_total"] = int(len(y_syn))
    meta["n_syn_post_total"] = int(len(y_syn))
    meta["n_syn"] = int(len(y_syn))
    if guard_counts:
        meta["note"] = (
            f"Rare-class guard applied before {mode}: quota=0 for classes {guard_counts} "
            f"with fewer than {int(args.min_real_for_synthesis)} real samples in D_tr."
        )

    w_aug = np.concatenate([
        np.ones(n_real, dtype=np.float32),
        np.full(len(y_syn), float(args.w_syn), dtype=np.float32),
    ])

    if save_synth and synth_out_dir:
        try:
            save_synth_bundle(
                out_dir=synth_out_dir,
                dataset_id=dataset_id,
                fold_idx=fold_idx,
                mode=mode,
                X_real=X_real,
                y_real=y_real,
                X_syn=X_syn,
                y_syn=y_syn,
                feature_names=feature_names,
                meta=meta,
            )
        except Exception:
            pass

    return AugmentationResult(
        X_aug=X_res,
        y_aug=y_res,
        w_aug=w_aug,
        X_syn=X_syn,
        y_syn=y_syn,
        meta=meta,
    )


def augment_training_set_by_mode(
    mode: str,
    dataset_id: str,
    fold_idx: int,
    X_tr_fs: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    args: Args,
    synth_out_dir: Optional[str],
    save_synth: bool,
) -> AugmentationResult:
    if is_diffusion_mode(mode):
        return augment_training_set(
            dataset_id=dataset_id,
            fold_idx=fold_idx,
            X_tr_fs=X_tr_fs,
            y_tr=y_tr,
            feature_names=feature_names,
            args=args,
            synth_out_dir=synth_out_dir,
            save_synth=save_synth,
            mode_label=str(mode),
        )
    if mode in {"SMOTE", "BorderlineSMOTE", "KMeansSMOTE"}:
        return augment_training_set_smote_family(
            mode=mode,
            dataset_id=dataset_id,
            fold_idx=fold_idx,
            X_tr_fs=X_tr_fs,
            y_tr=y_tr,
            feature_names=feature_names,
            args=args,
            synth_out_dir=synth_out_dir,
            save_synth=save_synth,
        )
    raise ValueError(f"Unsupported augmentation mode: {mode}")


def save_synth_bundle(
    out_dir: str,
    dataset_id: str,
    fold_idx: int,
    mode: str,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    feature_names: List[str],
    meta: Dict[str, Any],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{dataset_id}__fold{fold_idx:03d}__aug{mode}__nreal{len(y_real)}__nsyn{len(y_syn)}"
    npz_path = os.path.join(out_dir, stem + ".npz")
    np.savez_compressed(
        npz_path,
        X_real=X_real.astype(np.float32, copy=False),
        y_real=y_real.astype(np.int64, copy=False),
        X_syn=X_syn.astype(np.float32, copy=False),
        y_syn=y_syn.astype(np.int64, copy=False),
        feature_names=np.array(feature_names, dtype=object),
        meta=np.array([meta], dtype=object),
    )


def augment_training_set(
    dataset_id: str,
    fold_idx: int,
    X_tr_fs: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    args: Args,
    synth_out_dir: Optional[str],
    save_synth: bool,
    mode_label: str = "DiffAug",
) -> AugmentationResult:
    lambda_js_eff, lambda_mmd_eff = resolve_diffusion_lambdas(mode_label, args)
    real_counts = {int(k): int(v) for k, v in Counter(np.asarray(y_tr, dtype=int).tolist()).items()}
    quota, quota_meta = compute_minority_only_quota(
        y_tr,
        gamma=args.gamma,
        min_real_for_synthesis=args.min_real_for_synthesis,
    )

    guard_counts = {int(k): int(v) for k, v in quota_meta.get("guarded_class_counts", {}).items()}
    guard_note = ""
    if guard_counts:
        guard_note = (
            f"Rare-class guard applied: quota=0 for classes {guard_counts} "
            f"with fewer than {int(args.min_real_for_synthesis)} real samples in D_tr."
        )

    meta: Dict[str, Any] = {
        "mode": str(mode_label),
        "use_synthesis": not bool(args.no_diffusion),
        "use_hdbscan": not bool(args.no_hdbscan),
        "use_anchor": not bool(args.no_anchor),
        "filter_knn": not bool(args.no_knn),
        "knn_k": int(args.knn_k),
        "keep_q": float(args.keep_q),
        "gamma": float(args.gamma),
        "w_syn": float(args.w_syn),
        "min_real_for_synthesis": int(args.min_real_for_synthesis),
        "quota_target": int(quota_meta.get("target", 0)),
        "real_counts": real_counts,
        "quota": {int(k): int(v) for k, v in quota.items()},
        "guarded_classes": [int(x) for x in quota_meta.get("guarded_classes", [])],
        "guarded_class_counts": guard_counts,
        "syn_pre_counts": {},
        "syn_post_counts": {},
        "n_syn_pre_total": 0,
        "n_syn_post_total": 0,
        "n_syn": 0,
        "note": guard_note,
    }

    if args.no_diffusion:
        meta["note"] = ("Diffusion disabled by CLI. " + guard_note).strip()
        return AugmentationResult(
            X_aug=X_tr_fs,
            y_aug=np.asarray(y_tr, dtype=np.int64),
            w_aug=np.ones(len(y_tr), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    if sum(int(v) for v in quota.values()) <= 0:
        base_note = "No positive quota after minority-only allocation."
        meta["note"] = (base_note + " " + guard_note).strip()
        return AugmentationResult(
            X_aug=X_tr_fs,
            y_aug=np.asarray(y_tr, dtype=np.int64),
            w_aug=np.ones(len(y_tr), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    cfg = pick_diffusion_config(
        len(y_tr),
        alignment_bins=args.alignment_bins,
        seed=args.seed + 10000 + int(fold_idx),
        lambda_js=lambda_js_eff,
        lambda_mmd=lambda_mmd_eff,
    )
    model_g, a_cum = train_diffusion_on_fold(X_tr_fs, y_tr, cfg)

    X_syn, y_syn, gen_stats = generate_candidates_and_filter(
        model=model_g,
        X_tr=X_tr_fs,
        y_tr=y_tr,
        quota=quota,
        a_cum=a_cum,
        cfg=cfg,
        use_hdbscan=(not args.no_hdbscan),
        use_anchor=(not args.no_anchor),
        filter_knn=(not args.no_knn),
        knn_k=args.knn_k,
        keep_q=args.keep_q,
    )

    del model_g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta.update(gen_stats)

    if X_syn is None or y_syn is None or len(y_syn) == 0:
        meta["note"] = "Post-filter synthetic set is empty."
        return AugmentationResult(
            X_aug=X_tr_fs,
            y_aug=np.asarray(y_tr, dtype=np.int64),
            w_aug=np.ones(len(y_tr), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    meta["n_syn"] = int(len(y_syn))
    X_aug = np.vstack([X_tr_fs, X_syn])
    y_aug = np.concatenate([np.asarray(y_tr, dtype=np.int64), np.asarray(y_syn, dtype=np.int64)])
    w_aug = np.concatenate([
        np.ones(len(y_tr), dtype=np.float32),
        np.full(len(y_syn), float(args.w_syn), dtype=np.float32),
    ])

    if save_synth and synth_out_dir:
        try:
            save_synth_bundle(
                out_dir=synth_out_dir,
                dataset_id=dataset_id,
                fold_idx=fold_idx,
                mode=str(mode_label),
                X_real=X_tr_fs,
                y_real=np.asarray(y_tr, dtype=np.int64),
                X_syn=np.asarray(X_syn, dtype=np.float32),
                y_syn=np.asarray(y_syn, dtype=np.int64),
                feature_names=feature_names,
                meta=meta,
            )
        except Exception:
            pass

    return AugmentationResult(
        X_aug=X_aug,
        y_aug=y_aug,
        w_aug=w_aug,
        X_syn=X_syn,
        y_syn=y_syn,
        meta=meta,
    )


# =============================================================================
# POST-FILTER ALIGNMENT DIAGNOSTICS
# =============================================================================
def fixed_soft_hist_np(x: np.ndarray, centers: np.ndarray, width: np.ndarray) -> np.ndarray:
    diff = (x[:, :, None] - centers[None, :, :]) / width[None, :, None]
    h = np.exp(-0.5 * diff ** 2).sum(axis=0)
    h = h / (h.sum(axis=1, keepdims=True) + 1e-8)
    return h


def js_divergence_np(x_real: np.ndarray, x_syn: np.ndarray, bins: int) -> float:
    X = np.asarray(x_real, dtype=np.float32)
    Y = np.asarray(x_syn, dtype=np.float32)
    minv = X.min(axis=0)
    maxv = X.max(axis=0)
    span = maxv - minv
    pad = 0.05 * span + 1e-3
    minv = minv - pad
    maxv = maxv + pad
    same = span < 1e-6
    minv[same] = minv[same] - 0.5
    maxv[same] = maxv[same] + 0.5
    t = np.linspace(0.0, 1.0, bins, dtype=np.float32)[None, :]
    centers = minv[:, None] + (maxv - minv)[:, None] * t
    width = np.maximum((maxv - minv) / max(1, bins - 1), 1e-3).astype(np.float32)

    p = fixed_soft_hist_np(X, centers, width)
    q = fixed_soft_hist_np(Y, centers, width)
    m = 0.5 * (p + q)
    kl_p = (p * (np.log(p + 1e-8) - np.log(m + 1e-8))).sum(axis=1)
    kl_q = (q * (np.log(q + 1e-8) - np.log(m + 1e-8))).sum(axis=1)
    return float(0.5 * (kl_p + kl_q).mean())


def mmd_rbf_np(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    sigma = float(max(sigma, 1e-3))
    xx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    yy = ((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    xy = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
    denom = 2.0 * (sigma ** 2)
    k_xx = np.exp(-xx / denom).mean()
    k_yy = np.exp(-yy / denom).mean()
    k_xy = np.exp(-xy / denom).mean()
    return float(k_xx + k_yy - 2.0 * k_xy)


def compute_alignment_diagnostics(
    dataset_id: str,
    fold_idx: int,
    mode: str,
    X_real: np.ndarray,
    y_real: np.ndarray,
    X_syn: Optional[np.ndarray],
    y_syn: Optional[np.ndarray],
    bins: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if X_syn is None or y_syn is None or len(y_syn) == 0:
        return rows

    yr = np.asarray(y_real, dtype=int)
    ys = np.asarray(y_syn, dtype=int)

    js_vals = []
    mmd_vals = []
    for c in sorted(set(np.unique(yr).tolist()) & set(np.unique(ys).tolist())):
        xr = np.asarray(X_real[yr == int(c)], dtype=np.float32)
        xs = np.asarray(X_syn[ys == int(c)], dtype=np.float32)
        if len(xr) == 0 or len(xs) == 0:
            continue
        js = js_divergence_np(xr, xs, bins=bins)
        sigma = median_heuristic_sigma(xr)
        mmd = mmd_rbf_np(xr, xs, sigma=sigma)
        js_vals.append(js)
        mmd_vals.append(mmd)
        rows.append({
            "Dataset": dataset_id,
            "Fold": int(fold_idx),
            "Mode": mode,
            "Class": int(c),
            "n_real": int(len(xr)),
            "n_syn": int(len(xs)),
            "JS": float(js),
            "MMD": float(mmd),
            "sigma": float(sigma),
            "Level": "class",
        })

    if js_vals:
        rows.append({
            "Dataset": dataset_id,
            "Fold": int(fold_idx),
            "Mode": mode,
            "Class": "mean",
            "n_real": int(len(y_real)),
            "n_syn": int(len(y_syn)),
            "JS": float(np.mean(js_vals)),
            "MMD": float(np.mean(mmd_vals)),
            "sigma": np.nan,
            "Level": "fold_mean",
        })
    return rows


# =============================================================================
# BASELINES
# =============================================================================
def fit_predict_baseline(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    if model_name == "SVM":
        clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight=None)
    elif model_name == "SVM-balanced":
        clf = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced")
    elif model_name == "RF":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "GB":
        clf = GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    elif model_name == "XGB":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Install it or remove XGB from --models.")
        kwargs = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        )
        if num_classes > 2:
            clf = XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                **kwargs,
            )
        else:
            clf = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                **kwargs,
            )
    else:
        raise ValueError(f"Unsupported baseline model: {model_name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return y_pred.astype(np.int64, copy=False)


# =============================================================================
# KAN LAZY IMPORTS
# =============================================================================
_KAN_IMPORT_CACHE: Optional[Dict[str, Any]] = None


def add_extra_python_paths(paths: Sequence[str]) -> None:
    for p in paths:
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def get_kan_classes(extra_paths: Sequence[str]) -> Dict[str, Any]:
    global _KAN_IMPORT_CACHE
    if _KAN_IMPORT_CACHE is not None:
        return _KAN_IMPORT_CACHE

    add_extra_python_paths(extra_paths)

    errors: Dict[str, str] = {}
    out: Dict[str, Any] = {}

    try:
        from efficient_kan import KAN as EfficientKANClass
        out["EfficientKAN"] = EfficientKANClass
    except Exception as exc:
        errors["EfficientKAN"] = repr(exc)

    try:
        from fastkan import FastKAN as FastKANClass
        out["FastKAN"] = FastKANClass
    except Exception as exc:
        errors["FastKAN"] = repr(exc)

    try:
        from fasterkan import FasterKAN as FasterKANClass
        out["FasterKAN"] = FasterKANClass
    except Exception as exc:
        errors["FasterKAN"] = repr(exc)

    if len(out) == 0:
        msg = (
            "No KAN library could be imported. "
            "Install/import efficient_kan, fastkan, and fasterkan, or pass --extra_python_path.\n"
            f"Import errors: {errors}"
        )
        raise ImportError(msg)

    _KAN_IMPORT_CACHE = out
    return out


def build_kan_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    extra_paths: Sequence[str],
) -> torch.nn.Module:
    classes = get_kan_classes(extra_paths)

    if model_name == "EfficientKAN":
        cls = classes.get("EfficientKAN")
        if cls is None:
            raise ImportError("EfficientKAN class is unavailable in the current environment.")
        return cls(
            layers_hidden=[input_dim, 128, 64, num_classes],
            grid_size=8,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-7.0, 7.0],
        ).to(DEVICE)

    if model_name == "FastKAN":
        cls = classes.get("FastKAN")
        if cls is None:
            raise ImportError("FastKAN class is unavailable in the current environment.")
        return cls(
            layers_hidden=[input_dim, 128, 64, num_classes],
            grid_min=-7.0,
            grid_max=7.0,
            num_grids=8,
            use_base_update=True,
            base_activation=torch.nn.functional.silu,
            spline_weight_init_scale=0.1,
        ).to(DEVICE)

    if model_name == "FasterKAN":
        cls = classes.get("FasterKAN")
        if cls is None:
            raise ImportError("FasterKAN class is unavailable in the current environment.")
        return cls(
            layers_hidden=[input_dim, 128, 64, num_classes],
            grid_min=-7.0,
            grid_max=7.0,
            num_grids=8,
            exponent=2,
            inv_denominator=0.5,
            train_grid=True,
            train_inv_denominator=True,
            base_activation=torch.nn.SiLU(),
            spline_weight_init_scale=1.0,
        ).to(DEVICE)

    raise ValueError(f"Unsupported KAN model: {model_name}")


_KAN_PARAM_COUNT_CACHE: Dict[Tuple[str, int, int, Tuple[str, ...]], int] = {}
_MLP_MATCH_CACHE: Dict[Tuple[str, int, int, float], Dict[str, Any]] = {}


def get_kan_parameter_count(
    model_name: str,
    input_dim: int,
    num_classes: int,
    extra_paths: Sequence[str],
) -> int:
    key = (str(model_name), int(input_dim), int(num_classes), tuple(str(x) for x in extra_paths))
    if key in _KAN_PARAM_COUNT_CACHE:
        return int(_KAN_PARAM_COUNT_CACHE[key])

    model = build_kan_model(
        model_name=str(model_name),
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        extra_paths=extra_paths,
    )
    n_params = count_trainable_parameters(model)
    _KAN_PARAM_COUNT_CACHE[key] = int(n_params)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return int(n_params)


def build_torch_model(
    model_name: str,
    input_dim: int,
    num_classes: int,
    args: Args,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if model_name in KAN_MODEL_NAMES:
        model = build_kan_model(
            model_name=model_name,
            input_dim=input_dim,
            num_classes=num_classes,
            extra_paths=args.extra_python_path,
        )
        n_params = count_trainable_parameters(model)
        info = {
            "ModelFamily": "KAN",
            "MatchType": "kan_default",
            "HiddenDims": [128, 64],
            "n_trainable_params": int(n_params),
            "target_trainable_params": int(n_params),
            "param_match_abs_diff": 0,
            "param_match_rel_diff": 0.0,
            "param_match_target": model_name,
        }
        return model, info

    if model_name == "MLPShape":
        hidden_dims = [int(x) for x in args.mlp_shape_hidden]
        model = PlainMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            activation=args.mlp_activation,
            dropout=args.mlp_dropout,
        ).to(DEVICE)
        n_params = count_trainable_parameters(model)
        info = {
            "ModelFamily": "MLP",
            "MatchType": "shape_matched",
            "HiddenDims": hidden_dims,
            "n_trainable_params": int(n_params),
            "target_trainable_params": np.nan,
            "param_match_abs_diff": np.nan,
            "param_match_rel_diff": np.nan,
            "param_match_target": "",
        }
        return model, info

    if model_name in MLP_PARAM_TARGET_MAP:
        target_backbone = MLP_PARAM_TARGET_MAP[model_name]
        cache_key = (str(target_backbone), int(input_dim), int(num_classes), float(args.mlp_param_match_ratio))
        if cache_key not in _MLP_MATCH_CACHE:
            target_params = get_kan_parameter_count(
                model_name=target_backbone,
                input_dim=input_dim,
                num_classes=num_classes,
                extra_paths=args.extra_python_path,
            )
            hidden_dims, match_stats = solve_ratio_preserving_two_hidden_mlp(
                input_dim=input_dim,
                num_classes=num_classes,
                target_params=target_params,
                ratio=args.mlp_param_match_ratio,
            )
            _MLP_MATCH_CACHE[cache_key] = {
                "target_backbone": target_backbone,
                "target_params": int(target_params),
                "hidden_dims": [int(x) for x in hidden_dims],
                **match_stats,
            }

        match_stats = dict(_MLP_MATCH_CACHE[cache_key])
        hidden_dims = [int(x) for x in match_stats["hidden_dims"]]
        model = PlainMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            activation=args.mlp_activation,
            dropout=args.mlp_dropout,
        ).to(DEVICE)
        n_params = count_trainable_parameters(model)
        info = {
            "ModelFamily": "MLP",
            "MatchType": "param_matched",
            "HiddenDims": hidden_dims,
            "n_trainable_params": int(n_params),
            "target_trainable_params": int(match_stats["target_params"]),
            "param_match_abs_diff": int(abs(n_params - int(match_stats["target_params"]))),
            "param_match_rel_diff": float(abs(n_params - int(match_stats["target_params"])) / max(1, int(match_stats["target_params"]))),
            "param_match_target": str(target_backbone),
        }
        return model, info

    raise ValueError(f"Unsupported torch backbone: {model_name}")


# =============================================================================
# KAN TRAINING / INNER-CV EPOCH SELECTION
# =============================================================================
def evaluate_torch_model(model: torch.nn.Module, loader: DataLoader, num_classes: int) -> Dict[str, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss(reduction="sum")
    y_true: List[int] = []
    y_pred: List[int] = []
    loss_sum = 0.0
    n_sum = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            loss_sum += float(ce(out, yb).item())
            n_sum += int(yb.numel())
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    metrics = fixed_label_metrics(y_true_np, y_pred_np, num_classes=num_classes)
    metrics["ValLoss"] = float(loss_sum / max(1, n_sum))
    return metrics


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> None:
    model.train()
    ce = torch.nn.CrossEntropyLoss(reduction="none")
    for xb, yb, wb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        wb = wb.to(DEVICE)
        optimizer.zero_grad()
        out = model(xb)
        loss = (ce(out, yb) * wb).mean()
        loss.backward()
        optimizer.step()


def train_kan_with_real_val(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
) -> Tuple[int, float, str]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_f1 = False
    y_val_all: List[int] = []
    for _, yb in val_loader:
        y_val_all.extend(yb.numpy().tolist())
    if macro_f1_defined(np.asarray(y_val_all, dtype=int), num_classes):
        use_f1 = True

    best_score = -1e18
    best_state = None
    best_epoch = 1
    wait = 0

    for epoch in range(1, max_epochs + 1):
        train_one_epoch(model, train_loader, optimizer)
        val_metrics = evaluate_torch_model(model, val_loader, num_classes)
        score = float(val_metrics["F1"]) if use_f1 else float(-val_metrics["ValLoss"])

        if score > best_score + 1e-8:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if use_f1:
        return int(best_epoch), float(best_score), "macroF1"
    return int(best_epoch), float(-best_score), "valLoss"


def train_kan_fixed_epochs(
    model: torch.nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = max(1, int(epochs))
    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer)


def select_epoch_by_inner_cv(
    model_name: str,
    X_real: np.ndarray,
    y_real: np.ndarray,
    feature_names: List[str],
    args: Args,
    aug_mode: Optional[str],
) -> Tuple[int, str]:
    """
    When D_val is infeasible, select epoch by inner CV on D_tr only.
    The validation part of each inner fold is always real-only.
    For DiffAug, synthetic samples are generated only from the inner training split.
    """
    y_real = np.asarray(y_real, dtype=np.int64)
    if len(y_real) < 4 or len(np.unique(y_real)) < 2:
        return int(args.num_epochs), "fixed_fallback"

    inner_cv, inner_cv_name = choose_size_aware_cv(y_real, seed=args.seed + 777)
    epoch_scores: List[List[float]] = [[] for _ in range(int(args.num_epochs))]

    num_classes = int(np.max(y_real)) + 1
    fold_counter = 0

    for inner_fold_idx, (inner_tr_idx, inner_val_idx) in enumerate(inner_cv.split(X_real, y_real)):
        if len(inner_tr_idx) == 0 or len(inner_val_idx) == 0:
            continue

        X_in_tr = X_real[inner_tr_idx]
        y_in_tr = y_real[inner_tr_idx]
        X_in_val = X_real[inner_val_idx]
        y_in_val = y_real[inner_val_idx]

        if len(np.unique(y_in_tr)) < 2:
            continue

        if aug_mode is not None and aug_mode != "NoAug":
            aug_res = augment_training_set_by_mode(
                mode=str(aug_mode),
                dataset_id="innercv",
                fold_idx=int(inner_fold_idx),
                X_tr_fs=X_in_tr,
                y_tr=y_in_tr,
                feature_names=feature_names,
                args=args,
                synth_out_dir=None,
                save_synth=False,
            )
            X_train = aug_res.X_aug
            y_train = aug_res.y_aug
            w_train = aug_res.w_aug
        else:
            X_train = X_in_tr
            y_train = y_in_tr
            w_train = np.ones(len(y_in_tr), dtype=np.float32)

        train_loader = make_loader_with_weights(
            X_train, y_train, w_train, batch_size=args.batch_size, shuffle=True,
            seed=args.seed + 1000 + inner_fold_idx
        )
        val_loader = make_eval_loader(X_in_val, y_in_val, batch_size=args.batch_size)

        model, _ = build_torch_model(
            model_name=model_name,
            input_dim=int(X_real.shape[1]),
            num_classes=num_classes,
            args=args,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        use_f1 = macro_f1_defined(y_in_val, num_classes)
        for epoch in range(1, args.num_epochs + 1):
            train_one_epoch(model, train_loader, optimizer)
            val_metrics = evaluate_torch_model(model, val_loader, num_classes)
            score = float(val_metrics["F1"]) if use_f1 else float(-val_metrics["ValLoss"])
            epoch_scores[epoch - 1].append(score)

        fold_counter += 1

    if fold_counter == 0:
        return int(args.num_epochs), "fixed_fallback"

    mean_scores = np.array([
        np.mean(s) if len(s) > 0 else -1e18
        for s in epoch_scores
    ], dtype=np.float64)

    best_epoch = int(np.argmax(mean_scores)) + 1
    return best_epoch, f"innerCV:{inner_cv_name}"


# =============================================================================
# RESULTS WRITING / STATS
# =============================================================================
def summarize_group(df: pd.DataFrame, by_cols: List[str], metric_cols: List[str]) -> pd.DataFrame:
    agg = {}
    for col in metric_cols:
        agg[col] = ["mean", "std"]
    out = df.groupby(by_cols, dropna=False).agg(agg).reset_index()
    out.columns = [
        "_".join([c for c in tup if c]).rstrip("_")
        if isinstance(tup, tuple) else str(tup)
        for tup in out.columns
    ]
    return out


def compute_paired_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    work = summary_df.copy()
    work["Method"] = work["Model"] + "|" + work["Mode"]

    # Core manuscript comparisons + optional augmentation baselines
    target_pairs = [
        ("EfficientKAN|DiffAug", "EfficientKAN|NoAug"),
        ("FastKAN|DiffAug", "FastKAN|NoAug"),
        ("FasterKAN|DiffAug", "FasterKAN|NoAug"),
        ("EfficientKAN|DiffAug-noAlign", "EfficientKAN|NoAug"),
        ("FastKAN|DiffAug-noAlign", "FastKAN|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "FasterKAN|NoAug"),
        ("FasterKAN|DiffAug", "GB|NoAug"),
        ("FasterKAN|DiffAug", "XGB|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "GB|NoAug"),
        ("FasterKAN|DiffAug-noAlign", "XGB|NoAug"),
    ]
    for backbone in ["EfficientKAN", "FastKAN", "FasterKAN"]:
        for aug in ["SMOTE", "BorderlineSMOTE", "KMeansSMOTE"]:
            target_pairs.append((f"{backbone}|{aug}", f"{backbone}|NoAug"))
            target_pairs.append((f"{backbone}|DiffAug", f"{backbone}|{aug}"))
            target_pairs.append((f"{backbone}|DiffAug-noAlign", f"{backbone}|{aug}"))

    for a, b in target_pairs:
        da = work[work["Method"] == a][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "A"})
        db = work[work["Method"] == b][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "B"})
        merged = da.merge(db, on="Dataset", how="inner")
        if merged.shape[0] == 0:
            continue
        diffs = merged["A"].values - merged["B"].values
        if np.allclose(diffs, 0.0):
            p_value = 1.0
            stat = 0.0
        else:
            try:
                stat, p_value = wilcoxon(merged["A"].values, merged["B"].values, alternative="two-sided", zero_method="wilcox")
            except Exception:
                stat, p_value = np.nan, np.nan

        rows.append({
            "Comparison": f"{a} vs {b}",
            "Method_A": a,
            "Method_B": b,
            "n_datasets": int(merged.shape[0]),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_stat": float(stat) if stat == stat else np.nan,
            "p_value": float(p_value) if p_value == p_value else np.nan,
        })

    return pd.DataFrame(rows)


def compute_backbone_fairness_tests(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if summary_df.empty:
        return pd.DataFrame(rows)

    work = summary_df.copy()
    work["Method"] = work["Model"] + "|" + work["Mode"]
    target_pairs = [
        ("EfficientKAN|NoAug", "MLPShape|NoAug"),
        ("FastKAN|NoAug", "MLPShape|NoAug"),
        ("FasterKAN|NoAug", "MLPShape|NoAug"),
        ("EfficientKAN|NoAug", "MLPParam_EfficientKAN|NoAug"),
        ("FastKAN|NoAug", "MLPParam_FastKAN|NoAug"),
        ("FasterKAN|NoAug", "MLPParam_FasterKAN|NoAug"),
    ]

    for a, b in target_pairs:
        da = work[work["Method"] == a][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "A"})
        db = work[work["Method"] == b][["Dataset", "F1_mean"]].rename(columns={"F1_mean": "B"})
        merged = da.merge(db, on="Dataset", how="inner")
        if merged.shape[0] == 0:
            continue
        diffs = merged["A"].values - merged["B"].values
        if np.allclose(diffs, 0.0):
            stat = 0.0
            p_value = 1.0
        else:
            try:
                stat, p_value = wilcoxon(merged["A"].values, merged["B"].values, alternative="two-sided", zero_method="wilcox")
            except Exception:
                stat, p_value = np.nan, np.nan

        rows.append({
            "Comparison": f"{a} vs {b}",
            "Method_A": a,
            "Method_B": b,
            "n_datasets": int(merged.shape[0]),
            "mean_diff": float(np.mean(diffs)),
            "median_diff": float(np.median(diffs)),
            "wilcoxon_stat": float(stat) if stat == stat else np.nan,
            "p_value": float(p_value) if p_value == p_value else np.nan,
        })

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = os.path.join(args.results_path, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    fs_cache_dir = os.path.join(run_dir, "_fs_cache") if not args.no_fs else None
    synth_out_dir = os.path.join(run_dir, "_synth_cache") if args.save_synthesis else None
    artifacts_out_dir = os.path.join(run_dir, "_fold_artifacts") if args.save_fold_artifacts else None

    log_file = args.log_file if args.log_file.strip() else os.path.join(run_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    logger = TxtLogger(log_file, also_print=True)

    try:
        logger.write("=" * 100)
        logger.write("FoDiKAN paper-aligned run")
        logger.write(f"Device: {DEVICE}")
        logger.write(f"Run dir: {run_dir}")
        logger.write(f"Default data dir: {DEFAULT_DATA_DIR}")
        logger.write(f"Default results dir: {BASE_RESULTS_PATH}")
        logger.write(f"Args: {json.dumps(asdict(args), ensure_ascii=False, indent=2)}")
        logger.write("=" * 100)

        with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
            json.dump(asdict(args), fh, ensure_ascii=False, indent=2)

        items = discover_inputs(args.data_path, dataset_ids=args.dataset_ids)
        logger.write(f"[INPUT] data_path={args.data_path}")
        logger.write(f"[INPUT] discovered {len(items)} dataset(s)")
        if len(items) == 0:
            logger.write("[FATAL] No datasets found.")
            return

        fold_rows: List[Dict[str, Any]] = []
        align_rows: List[Dict[str, Any]] = []

        for item in items:
            dataset_id = str(item.dataset_id)
            logger.write("\n" + "#" * 100)
            logger.write(f"DATASET: {dataset_id}")
            logger.write("#" * 100)

            try:
                if item.kind == "dhkan":
                    X_all, y_all, feature_names = load_dataset_dhkan_style(item.path)
                    logger.write(f"[LOAD] kind=dhkan dir={item.path}")
                else:
                    X_all, y_all, feature_names = load_data_from_csv_smart(item.path)
                    logger.write(f"[LOAD] kind=csv file={item.path}")
            except Exception as exc:
                logger.write(f"[ERROR] Cannot read dataset {dataset_id}: {exc}")
                continue

            n_samples, n_features = X_all.shape
            num_classes = int(np.max(y_all)) + 1
            class_counts = {int(k): int(v) for k, v in Counter(y_all.tolist()).items()}
            logger.write(f"n={n_samples} | p={n_features} | classes={num_classes} | counts={class_counts}")

            outer_cv, cv_name = choose_size_aware_cv(y_all, seed=args.seed)
            logger.write(f"[CV] {cv_name}")

            requested_aug_modes = []
            for mode_name in args.modes:
                if mode_name == "NoAug":
                    continue
                if is_diffusion_mode(mode_name) and args.no_diffusion:
                    continue
                if mode_name in AUGMENT_MODES:
                    requested_aug_modes.append(mode_name)
            need_any_augmentation = bool(requested_aug_modes) and any(m in KAN_MODEL_NAMES for m in args.models)

            for fold_idx, (train_outer_idx, test_outer_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
                logger.write("\n" + "-" * 100)
                logger.write(f"Fold {fold_idx}")

                X_train_outer = X_all[train_outer_idx]
                y_train_outer = y_all[train_outer_idx]
                X_test_outer = X_all[test_outer_idx]
                y_test_outer = y_all[test_outer_idx]

                tr_rel, val_rel, split_meta = safe_train_val_split_indices(
                    y_train_outer, val_ratio=args.val_ratio, seed=args.seed + fold_idx
                )
                X_tr_raw = X_train_outer[tr_rel]
                y_tr = y_train_outer[tr_rel]
                if len(val_rel) > 0:
                    X_val_raw = X_train_outer[val_rel]
                    y_val = y_train_outer[val_rel]
                else:
                    X_val_raw = np.zeros((0, X_all.shape[1]), dtype=X_all.dtype)
                    y_val = np.zeros((0,), dtype=y_all.dtype)

                logger.write(
                    f"[SafeCV] |D_tr|={len(y_tr)} |D_val|={len(y_val)} |D_te|={len(y_test_outer)} "
                    f"| realized_val_ratio={split_meta['realized_val_ratio']:.4f} | reason={split_meta['reason']}"
                )
                logger.write(f"[SafeCV] D_tr counts={dict(Counter(y_tr.tolist()))}")
                if len(y_val) > 0:
                    logger.write(f"[SafeCV] D_val counts={dict(Counter(y_val.tolist()))}")
                logger.write(f"[SafeCV] D_te counts={dict(Counter(y_test_outer.tolist()))}")

                scaler = StandardScaler().fit(X_tr_raw)
                X_tr_std = scaler.transform(X_tr_raw)
                X_val_std = scaler.transform(X_val_raw) if len(y_val) else X_val_raw
                X_te_std = scaler.transform(X_test_outer)

                train_hash = hashlib.md5(np.asarray(train_outer_idx[tr_rel], dtype=np.int64).tobytes()).hexdigest()[:10]

                if not args.no_fs:
                    selector = BoMGeneSelector(
                        seed=args.seed,
                        cache_dir=fs_cache_dir,
                        pre_screen_topk=args.pre_screen_topk,
                        mrmr_candidate_pool=args.mrmr_candidate_pool,
                        mrmr_topk=args.mrmr_topk,
                        redundancy_threshold=args.mrmr_redundancy_threshold,
                        boruta_n_estimators=args.boruta_n_estimators,
                        boruta_max_iter=args.boruta_max_iter,
                        boruta_alpha=args.boruta_alpha,
                        min_keep=args.fs_min_keep,
                        fallback_topk=args.fs_fallback_topk,
                    )
                    selector.fit(
                        X=X_tr_std,
                        y=y_tr,
                        feature_names=feature_names,
                        dataset_id=dataset_id,
                        fold_idx=fold_idx,
                        train_hash=train_hash,
                    )
                    X_tr_fs = selector.transform(X_tr_std)
                    X_val_fs = selector.transform(X_val_std) if len(y_val) else X_val_std
                    X_te_fs = selector.transform(X_te_std)
                    kept_feature_names = selector.get_kept_names()
                    logger.write(
                        f"[FS] kept={len(kept_feature_names)} | mRMR={selector.stats_.n_mrmr} | "
                        f"Boruta={selector.stats_.n_boruta} | cache={selector.stats_.from_cache}"
                    )
                else:
                    X_tr_fs = X_tr_std
                    X_val_fs = X_val_std
                    X_te_fs = X_te_std
                    kept_feature_names = list(feature_names)
                    logger.write(f"[FS] disabled -> using all standardized features ({X_tr_fs.shape[1]})")

                # Prepare requested augmenters once per fold (reused across KAN backbones)
                outer_aug_by_mode: Dict[str, AugmentationResult] = {}
                if need_any_augmentation:
                    for aug_mode in requested_aug_modes:
                        aug_res = augment_training_set_by_mode(
                            mode=aug_mode,
                            dataset_id=dataset_id,
                            fold_idx=fold_idx,
                            X_tr_fs=X_tr_fs,
                            y_tr=y_tr,
                            feature_names=kept_feature_names,
                            args=args,
                            synth_out_dir=synth_out_dir,
                            save_synth=args.save_synthesis,
                        )
                        outer_aug_by_mode[str(aug_mode)] = aug_res
                        logger.write(
                            f"[{aug_mode}] note={aug_res.meta.get('note', '')} | "
                            f"n_syn={aug_res.meta.get('n_syn', 0)} | quota={aug_res.meta.get('quota', {})} | "
                            f"guarded={aug_res.meta.get('guarded_class_counts', {})}"
                        )
                        if aug_res.X_syn is not None and aug_res.y_syn is not None and len(aug_res.y_syn) > 0:
                            align_rows.extend(
                                compute_alignment_diagnostics(
                                    dataset_id=dataset_id,
                                    fold_idx=fold_idx,
                                    mode=str(aug_mode),
                                    X_real=X_tr_fs,
                                    y_real=y_tr,
                                    X_syn=aug_res.X_syn,
                                    y_syn=aug_res.y_syn,
                                    bins=args.alignment_bins,
                                )
                            )

                if artifacts_out_dir:
                    try:
                        os.makedirs(artifacts_out_dir, exist_ok=True)
                        primary_outer_aug_key = (
                            "DiffAug"
                            if "DiffAug" in outer_aug_by_mode
                            else ("DiffAug-noAlign" if "DiffAug-noAlign" in outer_aug_by_mode else None)
                        )
                        artifact = {
                            "dataset_id": dataset_id,
                            "fold": fold_idx,
                            "cv_name": cv_name,
                            "train_outer_size": int(len(train_outer_idx)),
                            "test_outer_size": int(len(test_outer_idx)),
                            "d_tr_size": int(len(y_tr)),
                            "d_val_size": int(len(y_val)),
                            "d_te_size": int(len(y_test_outer)),
                            "split_meta": split_meta,
                            "n_selected_features": int(X_tr_fs.shape[1]),
                            "selected_feature_names": kept_feature_names,
                            "outer_aug_meta": outer_aug_by_mode.get(primary_outer_aug_key).meta if primary_outer_aug_key else {},
                            "outer_aug_meta_by_mode": {k: v.meta for k, v in outer_aug_by_mode.items()},
                        }
                        with open(os.path.join(artifacts_out_dir, f"{dataset_id}_fold{fold_idx:03d}.json"), "w", encoding="utf-8") as fh:
                            json.dump(artifact, fh, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                # --------------------------
                # Run baselines, KANs, and MLP controls
                # --------------------------
                for model_name in args.models:
                    is_kan = model_name in KAN_MODEL_NAMES
                    is_torch_model = model_name in TORCH_MODEL_NAMES
                    modes_for_model = ["NoAug"]
                    if is_kan:
                        extra_modes = []
                        for mode_name in args.modes:
                            if mode_name == "NoAug":
                                continue
                            if is_diffusion_mode(mode_name) and args.no_diffusion:
                                continue
                            if mode_name in AUGMENT_MODES:
                                extra_modes.append(mode_name)
                        modes_for_model = list(dict.fromkeys(["NoAug"] + extra_modes))

                    for mode in modes_for_model:
                        if mode not in args.modes:
                            continue
                        logger.write(f"[RUN] model={model_name} | mode={mode}")
                        t0 = time.time()
                        aug_meta_for_row: Optional[Dict[str, Any]] = None
                        model_info: Dict[str, Any] = {
                            "ModelFamily": "Baseline",
                            "MatchType": "baseline",
                            "HiddenDims": [],
                            "n_trainable_params": np.nan,
                            "target_trainable_params": np.nan,
                            "param_match_abs_diff": np.nan,
                            "param_match_rel_diff": np.nan,
                            "param_match_target": "",
                        }

                        try:
                            if not is_torch_model:
                                y_pred = fit_predict_baseline(
                                    model_name=model_name,
                                    X_train=X_tr_fs,
                                    y_train=y_tr,
                                    X_test=X_te_fs,
                                    num_classes=num_classes,
                                    seed=args.seed + fold_idx,
                                )
                                metrics = fixed_label_metrics(y_test_outer, y_pred, num_classes=num_classes)
                                best_epoch = np.nan
                                val_criterion = "n/a"
                                n_syn_used = 0
                            else:
                                if is_kan and mode != "NoAug" and mode in outer_aug_by_mode:
                                    aug_res = outer_aug_by_mode[mode]
                                    X_train = aug_res.X_aug
                                    y_train = aug_res.y_aug
                                    w_train = aug_res.w_aug
                                    n_syn_used = int(aug_res.meta.get("n_syn", 0))
                                    aug_meta_for_row = aug_res.meta
                                else:
                                    X_train = X_tr_fs
                                    y_train = y_tr
                                    w_train = np.ones(len(y_tr), dtype=np.float32)
                                    n_syn_used = 0

                                train_loader = make_loader_with_weights(
                                    X_train, y_train, w_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    seed=args.seed + fold_idx,
                                )
                                test_loader = make_eval_loader(X_te_fs, y_test_outer, batch_size=args.batch_size)

                                model, model_info = build_torch_model(
                                    model_name=model_name,
                                    input_dim=int(X_tr_fs.shape[1]),
                                    num_classes=num_classes,
                                    args=args,
                                )
                                logger.write(
                                    f"[MODEL] family={model_info['ModelFamily']} | match={model_info['MatchType']} | "
                                    f"hidden={model_info['HiddenDims']} | params={model_info['n_trainable_params']} | "
                                    f"target={model_info['param_match_target']}:{model_info['target_trainable_params']}"
                                )

                                if len(y_val) > 0:
                                    val_loader = make_eval_loader(X_val_fs, y_val, batch_size=args.batch_size)
                                    best_epoch, _, val_criterion = train_kan_with_real_val(
                                        model=model,
                                        train_loader=train_loader,
                                        val_loader=val_loader,
                                        num_classes=num_classes,
                                        lr=1e-3,
                                        weight_decay=1e-4,
                                        max_epochs=args.num_epochs,
                                        patience=args.patience,
                                    )
                                else:
                                    best_epoch, val_criterion = select_epoch_by_inner_cv(
                                        model_name=model_name,
                                        X_real=X_tr_fs,
                                        y_real=y_tr,
                                        feature_names=kept_feature_names,
                                        args=args,
                                        aug_mode=(mode if (is_kan and mode != "NoAug") else None),
                                    )
                                    train_kan_fixed_epochs(
                                        model=model,
                                        train_loader=train_loader,
                                        epochs=best_epoch,
                                        lr=1e-3,
                                        weight_decay=1e-4,
                                    )

                                metrics = evaluate_torch_model(model, test_loader, num_classes)
                                metrics = {
                                    "Accuracy": float(metrics["Accuracy"]),
                                    "Precision": float(metrics["Precision"]),
                                    "Recall": float(metrics["Recall"]),
                                    "F1": float(metrics["F1"]),
                                }
                        except Exception as exc:
                            logger.write(f"[ERROR] model={model_name} | mode={mode} | {exc}")
                            continue

                        train_time_sec = float(time.time() - t0)

                        row = {
                            "Dataset": dataset_id,
                            "Model": model_name,
                            "Mode": mode,
                            "Fold": int(fold_idx),
                            "CV_Type": cv_name,
                            "n_samples": int(n_samples),
                            "n_features_raw": int(n_features),
                            "n_classes": int(num_classes),
                            "n_features_selected": int(X_tr_fs.shape[1]),
                            "n_syn": int(n_syn_used),
                            "ModelFamily": model_info["ModelFamily"],
                            "MatchType": model_info["MatchType"],
                            "HiddenDims": json.dumps(model_info["HiddenDims"], ensure_ascii=False),
                            "n_trainable_params": model_info["n_trainable_params"],
                            "target_trainable_params": model_info["target_trainable_params"],
                            "param_match_abs_diff": model_info["param_match_abs_diff"],
                            "param_match_rel_diff": model_info["param_match_rel_diff"],
                            "param_match_target": model_info["param_match_target"],
                            "d_tr_size": int(len(y_tr)),
                            "d_val_size": int(len(y_val)),
                            "d_te_size": int(len(y_test_outer)),
                            "realized_val_ratio": float(split_meta["realized_val_ratio"]),
                            "val_classes_missing": int(num_classes - len(np.unique(y_val))) if len(y_val) else int(num_classes),
                            "test_classes_missing": int(num_classes - len(np.unique(y_test_outer))),
                            "rare_guard_threshold": int(args.min_real_for_synthesis),
                            "rare_guarded_classes": json.dumps(aug_meta_for_row.get("guarded_class_counts", {}), ensure_ascii=False) if aug_meta_for_row is not None else "",
                            "BestEpoch": best_epoch,
                            "ValCriterion": val_criterion,
                            "TrainTime_sec": train_time_sec,
                            **metrics,
                        }
                        fold_rows.append(row)
                        logger.write(
                            f"[DONE] {model_name}/{mode} | F1={metrics['F1']:.4f} | "
                            f"Acc={metrics['Accuracy']:.4f} | nSyn={n_syn_used} | "
                            f"Params={model_info['n_trainable_params']} | BestEpoch={best_epoch} | ValCriterion={val_criterion}"
                        )

        fold_df = pd.DataFrame(fold_rows)
        if fold_df.empty:
            logger.write("[FATAL] No successful fold results were produced.")
            return

        metric_cols = [
            "n_features_selected", "n_syn",
            "n_trainable_params", "target_trainable_params", "param_match_abs_diff", "param_match_rel_diff",
            "Accuracy", "Precision", "Recall", "F1",
            "TrainTime_sec",
        ]
        summary_df = summarize_group(
            fold_df,
            by_cols=["Dataset", "Model", "Mode", "CV_Type", "n_samples", "n_features_raw", "n_classes"],
            metric_cols=metric_cols,
        )
        summary_df = summary_df.rename(columns={
            "n_features_selected_mean": "n_features_mean",
            "n_features_selected_std": "n_features_std",
            "n_syn_mean": "n_syn_mean",
            "n_syn_std": "n_syn_std",
            "n_trainable_params_mean": "n_trainable_params_mean",
            "n_trainable_params_std": "n_trainable_params_std",
            "target_trainable_params_mean": "target_trainable_params_mean",
            "target_trainable_params_std": "target_trainable_params_std",
            "param_match_abs_diff_mean": "param_match_abs_diff_mean",
            "param_match_abs_diff_std": "param_match_abs_diff_std",
            "param_match_rel_diff_mean": "param_match_rel_diff_mean",
            "param_match_rel_diff_std": "param_match_rel_diff_std",
            "Accuracy_mean": "Accuracy_mean",
            "Accuracy_std": "Accuracy_std",
            "Precision_mean": "Precision_mean",
            "Precision_std": "Precision_std",
            "Recall_mean": "Recall_mean",
            "Recall_std": "Recall_std",
            "F1_mean": "F1_mean",
            "F1_std": "F1_std",
            "TrainTime_sec_mean": "TrainTime_mean_sec",
            "TrainTime_sec_std": "TrainTime_std_sec",
        })

        paired_df = compute_paired_tests(summary_df)
        fairness_df = compute_backbone_fairness_tests(summary_df)
        align_df = pd.DataFrame(align_rows)

        fold_path = os.path.join(run_dir, "fold_details.csv")
        summary_path = os.path.join(run_dir, "dataset_summary.csv")
        paired_path = os.path.join(run_dir, "paired_tests.csv")
        fairness_path = os.path.join(run_dir, "backbone_fairness_tests.csv")
        align_path = os.path.join(run_dir, "alignment_diagnostics.csv")

        fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        paired_df.to_csv(paired_path, index=False, encoding="utf-8-sig")
        fairness_df.to_csv(fairness_path, index=False, encoding="utf-8-sig")
        align_df.to_csv(align_path, index=False, encoding="utf-8-sig")

        logger.write("\n" + "=" * 100)
        logger.write("RUN SUMMARY")
        logger.write("=" * 100)

        try:
            display_cols = ["Dataset", "Model", "Mode", "F1_mean", "F1_std", "Accuracy_mean", "n_features_mean", "n_syn_mean"]
            show = summary_df[display_cols].copy().sort_values(["Dataset", "Model", "Mode"]).reset_index(drop=True)
            rows = []
            for _, r in show.iterrows():
                rows.append([
                    r["Dataset"],
                    r["Model"],
                    r["Mode"],
                    f"{r['F1_mean']:.4f}",
                    f"{r['F1_std']:.4f}" if pd.notna(r["F1_std"]) else "nan",
                    f"{r['Accuracy_mean']:.4f}",
                    f"{r['n_features_mean']:.1f}",
                    f"{r['n_syn_mean']:.1f}",
                ])
            logger.write(_fmt_table(rows, headers=["Dataset", "Model", "Mode", "F1_mean", "F1_std", "Acc_mean", "nFeat_mean", "nSyn_mean"]))
        except Exception:
            pass

        if not paired_df.empty:
            logger.write("\n[Paired Wilcoxon tests]")
            rows = []
            for _, r in paired_df.iterrows():
                rows.append([
                    r["Comparison"],
                    str(int(r["n_datasets"])),
                    f"{r['mean_diff']:.4f}",
                    f"{r['median_diff']:.4f}",
                    f"{r['p_value']:.6f}" if pd.notna(r["p_value"]) else "nan",
                ])
            logger.write(_fmt_table(rows, headers=["Comparison", "n", "mean_diff", "median_diff", "p_value"]))

        logger.write("\nSaved files:")
        logger.write(f"  {fold_path}")
        logger.write(f"  {summary_path}")
        logger.write(f"  {paired_path}")
        logger.write(f"  {fairness_path}")
        logger.write(f"  {align_path}")
        logger.write(f"  {log_file}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
