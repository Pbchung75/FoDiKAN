"""Synthetic data generation and augmentation strategies."""

from __future__ import annotations
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from fodikan.config import Args, is_diffusion_mode, resolve_diffusion_lambdas
from fodikan.diffusion.model import DiffusionConfig, DiffusionDenoiser, get_beta_schedule, pick_diffusion_config, train_diffusion_on_fold
from fodikan.utils.repro import DEVICE

try:
    from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE, SMOTE

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


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

@torch.no_grad()
def ddpm_sample_from_noise(
    model: DiffusionDenoiser,
    n_samples: int,
    dim: int,
    y_cond: torch.Tensor,
    T: int,
) -> torch.Tensor:
    model.eval()
    betas, a_cum = get_beta_schedule(T)
    alphas = 1.0 - betas
    x_t = torch.randn((n_samples, dim), device=DEVICE)

    for t in reversed(range(T)):
        tt = torch.full((n_samples,), int(t), dtype=torch.long, device=DEVICE)
        eps = model(x_t, tt, y_cond, None)
        alpha_t = alphas[t]
        alpha_bar_t = a_cum[t]

        if t == 0:
            x_t = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t + 1e-8)
            continue

        alpha_bar_prev = a_cum[t - 1]
        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)
        mean = (x_t - coef * eps) / torch.sqrt(alpha_t + 1e-8)
        var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)) * (1.0 - alpha_t)
        noise = torch.randn_like(x_t)
        x_t = mean + torch.sqrt(torch.clamp(var, min=1e-12)) * noise

    return x_t

def augment_training_set_tabddpm(
    dataset_id: str,
    fold_idx: int,
    X_tr_fs: np.ndarray,
    y_tr: np.ndarray,
    feature_names: List[str],
    args: Args,
    synth_out_dir: Optional[str],
    save_synth: bool,
) -> AugmentationResult:
    X_real = np.asarray(X_tr_fs, dtype=np.float32)
    y_real = np.asarray(y_tr, dtype=np.int64)

    real_counts = {int(k): int(v) for k, v in Counter(y_real.tolist()).items()}
    quota, quota_meta = compute_minority_only_quota(
        y_real,
        gamma=1.0,
        min_real_for_synthesis=args.min_real_for_synthesis,
    )
    guard_counts = {int(k): int(v) for k, v in quota_meta.get("guarded_class_counts", {}).items()}

    meta: Dict[str, Any] = {
        "use_synthesis": True,
        "augmenter": "TabDDPM",
        "balancing": "to_max",
        "sampler": "ancestral_ddpm",
        "gamma": 1.0,
        "w_syn": float(args.w_syn),
        "min_real_for_synthesis": int(args.min_real_for_synthesis),
        "quota_target": int(quota_meta.get("target", 0)),
        "real_counts": real_counts,
        "quota": {int(k): int(v) for k, v in quota.items()},
        "sampling_strategy": _quota_to_sampling_strategy(y_real, quota),
        "guarded_classes": [int(x) for x in quota_meta.get("guarded_classes", [])],
        "guarded_class_counts": guard_counts,
        "syn_pre_counts": {},
        "syn_post_counts": {},
        "n_syn_pre_total": 0,
        "n_syn_post_total": 0,
        "n_syn": 0,
        "note": "",
    }

    if sum(int(v) for v in quota.values()) <= 0:
        meta["note"] = "No positive quota after to-max minority allocation."
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    cfg = pick_diffusion_config(
        len(y_real),
        alignment_bins=args.alignment_bins,
        seed=args.seed + 30000 + int(fold_idx),
        lambda_js=0.0,
        lambda_mmd=0.0,
    )
    cfg.self_condition = False

    model_g, _a_cum = train_diffusion_on_fold(X_real, y_real, cfg)

    parts_X: List[np.ndarray] = []
    parts_y: List[np.ndarray] = []
    for c, q in sorted(quota.items()):
        n_target = int(q)
        if n_target <= 0:
            continue
        remaining = n_target
        while remaining > 0:
            m = min(1024, remaining)
            y_cond = torch.full((m,), int(c), dtype=torch.long, device=DEVICE)
            x_gen = ddpm_sample_from_noise(
                model=model_g,
                n_samples=m,
                dim=int(X_real.shape[1]),
                y_cond=y_cond,
                T=cfg.T,
            )
            parts_X.append(x_gen.detach().cpu().numpy().astype(np.float32, copy=False))
            parts_y.append(np.full(m, int(c), dtype=np.int64))
            remaining -= m

    del model_g
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not parts_X:
        meta["note"] = "TabDDPM returned no synthetic samples on this fold."
        return AugmentationResult(
            X_aug=X_real,
            y_aug=y_real,
            w_aug=np.ones(len(y_real), dtype=np.float32),
            X_syn=None,
            y_syn=None,
            meta=meta,
        )

    X_syn = np.vstack(parts_X).astype(np.float32, copy=False)
    y_syn = np.concatenate(parts_y).astype(np.int64, copy=False)
    syn_counts = Counter(y_syn.tolist())
    meta["syn_pre_counts"] = {int(k): int(v) for k, v in syn_counts.items()}
    meta["syn_post_counts"] = {int(k): int(v) for k, v in syn_counts.items()}
    meta["n_syn_pre_total"] = int(len(y_syn))
    meta["n_syn_post_total"] = int(len(y_syn))
    meta["n_syn"] = int(len(y_syn))
    if guard_counts:
        meta["note"] = (
            f"Rare-class guard applied before TabDDPM: quota=0 for classes {guard_counts} "
            f"with fewer than {int(args.min_real_for_synthesis)} real samples in D_tr."
        )

    X_aug = np.vstack([X_real, X_syn])
    y_aug = np.concatenate([y_real, y_syn])
    w_aug = np.concatenate([
        np.ones(len(y_real), dtype=np.float32),
        np.full(len(y_syn), float(args.w_syn), dtype=np.float32),
    ])

    if save_synth and synth_out_dir:
        try:
            save_synth_bundle(
                out_dir=synth_out_dir,
                dataset_id=dataset_id,
                fold_idx=fold_idx,
                mode="TabDDPM",
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
        X_aug=X_aug,
        y_aug=y_aug,
        w_aug=w_aug,
        X_syn=X_syn,
        y_syn=y_syn,
        meta=meta,
    )

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
    gamma_eff = 1.0 if mode == "KMeansSMOTE" else float(args.gamma)
    quota, quota_meta = compute_minority_only_quota(
        y_real,
        gamma=gamma_eff,
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
        "gamma": float(gamma_eff),
        "balancing": "to_max" if mode == "KMeansSMOTE" else "gamma_overshoot",
        "w_syn": float(args.w_syn),
        "min_real_for_synthesis": int(args.min_real_for_synthesis),
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
    if mode == "TabDDPM":
        return augment_training_set_tabddpm(
            dataset_id=dataset_id,
            fold_idx=fold_idx,
            X_tr_fs=X_tr_fs,
            y_tr=y_tr,
            feature_names=feature_names,
            args=args,
            synth_out_dir=synth_out_dir,
            save_synth=save_synth,
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
