"""KAN backbone loading and construction."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Sequence

import torch

from fodikan.utils.repro import DEVICE


KAN_MODEL_NAMES = {"EfficientKAN", "FastKAN", "FasterKAN"}


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
