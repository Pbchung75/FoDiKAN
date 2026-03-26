"""Cross-validation and metric helpers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold


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

    alloc = {int(c): 1 for c in classes.tolist()}
    remaining = int(target - n_classes)

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
