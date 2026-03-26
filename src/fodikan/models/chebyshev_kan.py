"""Local ChebyshevKAN implementation used as a polynomial-family KAN comparator."""

from __future__ import annotations
from typing import Iterable, List, Sequence
import torch


class ChebyshevKANLayer(torch.nn.Module):
    """Edge-wise Chebyshev polynomial layer.

    Each input feature is first squashed into ``[-1, 1]`` and then expanded into
    Chebyshev polynomials of the first kind up to a fixed degree. The output is a
    learned weighted sum over all feature/degree combinations.
    """

    def __init__(self, in_features: int, out_features: int, degree: int = 5, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.degree = int(max(1, degree))

        self.coeff = torch.nn.Parameter(
            torch.empty(self.out_features, self.in_features, self.degree + 1)
        )
        self.base = torch.nn.Linear(self.in_features, self.out_features, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.coeff)
        torch.nn.init.xavier_uniform_(self.base.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"ChebyshevKANLayer expects a 2D tensor, got shape={tuple(x.shape)}")

        x_norm = torch.tanh(x)
        polys: List[torch.Tensor] = [torch.ones_like(x_norm), x_norm]
        for _ in range(2, self.degree + 1):
            polys.append(2.0 * x_norm * polys[-1] - polys[-2])
        cheb = torch.stack(polys[: self.degree + 1], dim=-1)
        y = torch.einsum("bid,oid->bo", cheb, self.coeff)
        y = y + self.base(x)
        if self.bias is not None:
            y = y + self.bias
        return y


class ChebyshevKAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden: Sequence[int],
        degree: int = 5,
        dropout: float = 0.0,
        input_normalization: str = "tanh",
    ) -> None:
        super().__init__()
        dims = [int(x) for x in layers_hidden]
        if len(dims) < 2:
            raise ValueError("layers_hidden must contain at least input and output dimensions.")
        self.degree = int(max(1, degree))
        self.input_normalization = str(input_normalization)

        layers: List[torch.nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(ChebyshevKANLayer(din, dout, degree=self.degree, bias=True))
            is_last = (i == len(dims) - 2)
            if not is_last:
                layers.append(torch.nn.LayerNorm(dout))
                layers.append(torch.nn.SiLU())
                if float(dropout) > 0.0:
                    layers.append(torch.nn.Dropout(float(dropout)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_normalization == "tanh":
            x = torch.tanh(x)
        elif self.input_normalization == "clamp":
            x = torch.clamp(x, min=-1.0, max=1.0)
        return self.net(x)
