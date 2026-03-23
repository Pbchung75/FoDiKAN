"""Training loops and epoch selection for KAN backbones."""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from fodikan.config import Args
from fodikan.data.cv import choose_size_aware_cv, fixed_label_metrics, macro_f1_defined
from fodikan.data.loaders import make_eval_loader, make_loader_with_weights
from fodikan.diffusion.augment import augment_training_set_by_mode
from fodikan.models.kan import build_kan_model
from fodikan.utils.repro import DEVICE


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

        model = build_kan_model(
            model_name=model_name,
            input_dim=int(X_real.shape[1]),
            num_classes=num_classes,
            extra_paths=args.extra_python_path,
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
