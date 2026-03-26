"""Diffusion model and alignment losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader, TensorDataset

from fodikan.utils.repro import DEVICE, set_seed


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
