from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from fodikan.utils.repro import DEVICE


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
