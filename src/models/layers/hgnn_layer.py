# -*- coding: utf-8 -*-
"""
Hyperbolic graph convolution layer (Lorentz manifold version), consistent with 
LorentzGraphConvolution in guoxinyu0617/MHR, adapted to this project's 
``from src.models.manifolds.lorentz import Lorentz`` interface.

Only implements attention-free decomposition (use_att=False) to avoid dependency 
on MHR's att_layers.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

from src.models.manifolds.lorentz import Lorentz


class LorentzLinear(nn.Module):
    """Linear transformation on Lorentz manifold (consistent with MHR LorentzLinear)."""

    def __init__(
        self,
        manifold: Lorentz,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.1,
        scale: float = 10.0,
        fixscale: bool = False,
        nonlin: Any | None = None,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, in_features], on Lorentz manifold (last column is time + space)
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(Module):
    """Lorentz neighborhood aggregation (attention-free, sparse adjacency matrix)."""

    def __init__(
        self,
        manifold: Lorentz,
        in_features: int,
        dropout: float,
        use_att: bool,
        local_agg: bool,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if use_att:
            raise NotImplementedError("Current hgnn_layer only supports use_att=False to reduce dependencies.")

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [num_nodes, in_features], adj: sparse [num_nodes, num_nodes]
        if adj.is_sparse:
            adj_t = adj if adj.dtype == x.dtype else adj.to(dtype=x.dtype)
            support_t = torch.sparse.mm(adj_t, x)
        else:
            support_t = adj @ x
        # Minkowski inner product used to reproject aggregated vectors back to hyperbolic space (same as MHR LorentzAgg)
        inner_self = self.manifold.inner(support_t, support_t, keepdim=True)
        denom = (-inner_self).abs().clamp_min(1e-8).sqrt()
        return support_t / denom


class HyperbolicGraphConvolution(nn.Module):
    """
    Lorentz hyperbolic graph convolution layer (interface name defined by project).

    Forward input is tuple ``(x, adj)``, consistent with LorentzGraphConvolution in MHR:
    - x: [num_nodes, manifold_dim] hyperbolic points
    - adj: sparse Float COO [num_nodes, num_nodes]
    """

    def __init__(
        self,
        manifold: Lorentz,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0.1,
        use_att: bool = False,
        local_agg: bool = True,
        nonlin: Any | None = None,
    ) -> None:
        super().__init__()
        self.manifold = manifold
        self.linear = LorentzLinear(
            manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin
        )
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att, local_agg)

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x, adj = input
        h = self.linear(x)
        h = self.agg(h, adj)
        return h, adj
