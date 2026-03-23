"""Optional MLP controls extracted from the backbone-fairness code path."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch


class PlainMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(num_classes)]
        if len(dims) < 2:
            raise ValueError("PlainMLP needs at least input and output dimensions.")

        layers: List[torch.nn.Module] = []
        act_name = str(activation).lower()
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            is_last = (i == len(dims) - 2)
            if not is_last:
                if act_name == "relu":
                    layers.append(torch.nn.ReLU())
                elif act_name == "gelu":
                    layers.append(torch.nn.GELU())
                elif act_name == "silu":
                    layers.append(torch.nn.SiLU())
                else:
                    raise ValueError(f"Unsupported MLP activation: {activation}")
                if float(dropout) > 0.0:
                    layers.append(torch.nn.Dropout(float(dropout)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def count_mlp_parameters(input_dim: int, hidden_dims: Sequence[int], num_classes: int) -> int:
    dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [int(num_classes)]
    total = 0
    for din, dout in zip(dims[:-1], dims[1:]):
        total += int(din) * int(dout) + int(dout)
    return int(total)

def solve_ratio_preserving_two_hidden_mlp(
    input_dim: int,
    num_classes: int,
    target_params: int,
    ratio: float = 0.5,
    min_width: int = 4,
    max_width: int = 8192,
) -> Tuple[List[int], Dict[str, Any]]:
    ratio = float(max(1e-3, ratio))
    target_params = int(max(1, target_params))
    min_width = int(max(1, min_width))
    max_width = int(max(min_width, max_width))

    def make_hidden(h1: int) -> List[int]:
        h1 = int(max(min_width, h1))
        h2 = int(max(min_width, round(h1 * ratio)))
        return [h1, h2]

    upper = int(max(min_width, 128))
    while count_mlp_parameters(input_dim, make_hidden(upper), num_classes) < target_params and upper < max_width:
        upper *= 2
    upper = int(min(upper, max_width))

    best_hidden = make_hidden(min_width)
    best_params = count_mlp_parameters(input_dim, best_hidden, num_classes)
    best_key = (
        abs(best_params - target_params) / max(1, target_params),
        abs(best_params - target_params),
        abs(best_hidden[0] - 128),
        abs(best_hidden[1] - 64),
    )

    for h1 in range(min_width, upper + 1):
        hidden = make_hidden(h1)
        params = count_mlp_parameters(input_dim, hidden, num_classes)
        key = (
            abs(params - target_params) / max(1, target_params),
            abs(params - target_params),
            abs(hidden[0] - 128),
            abs(hidden[1] - 64),
        )
        if key < best_key:
            best_hidden = hidden
            best_params = int(params)
            best_key = key

    return best_hidden, {
        "hidden_dims": [int(x) for x in best_hidden],
        "matched_params": int(best_params),
        "target_params": int(target_params),
        "abs_diff": int(abs(best_params - target_params)),
        "rel_diff": float(abs(best_params - target_params) / max(1, target_params)),
        "ratio": float(ratio),
        "search_upper": int(upper),
    }
