from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from fodikan.diffusion.model import median_heuristic_sigma


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
