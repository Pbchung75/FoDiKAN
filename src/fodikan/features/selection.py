"""Fold-local feature selection with BoMGene."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif


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
