# -*- coding: utf-8 -*-
"""
Lorentz manifold implementation (simple version for testing).

This is a simplified implementation for testing purposes.
For production, please run scripts/01_clone_repos.sh to get the full MHR implementation.
"""

import torch
import torch.nn as nn


class Lorentz(nn.Module):
    """
    Simplified Lorentz manifold implementation.
    
    Implements basic operations:
    - proju0: project to tangent space at origin
    - expmap0: exponential map from origin
    - projx: projection onto manifold
    - logmap0: logarithmic map to origin
    - inner: Minkowski inner product
    """

    def __init__(self, k: float = 1.0, learnable: bool = False):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(k), requires_grad=learnable)

    def proju0(self, u: torch.Tensor) -> torch.Tensor:
        """Project vector to tangent space at origin."""
        return u

    def expmap0(self, u: torch.Tensor, project: bool = True) -> torch.Tensor:
        """Exponential map from origin to manifold.
        
        输入 u 应该已经包含时间维度（第一列），形状为 [batch, d+1]
        """
        # 提取空间部分（去掉时间维度）
        d = u.size(-1) - 1
        u_space = u.narrow(-1, 1, d)
        
        # 计算范数和指数映射
        norm = torch.norm(u_space, dim=-1, keepdim=True).clamp_min(1e-8)
        coeff = torch.sinh(norm) / norm
        
        # 时间分量
        t = torch.cosh(norm)
        
        # 空间分量
        x = coeff * u_space
        
        # 合并时间和空间分量
        result = torch.cat([t, x], dim=-1)
        
        if project:
            return self.projx(result)
        return result

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """Project point onto Lorentz manifold."""
        # Ensure x[0]^2 - ||x[1:]||^2 = k
        d = x.size(-1) - 1
        t = x.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, d)
        
        # Compute squared norm
        norm_sq = torch.sum(x_narrow * x_narrow, dim=-1, keepdim=True)
        t_sq = t * t
        
        # Ensure hyperboloid constraint: t^2 - ||x||^2 = k
        scale = torch.sqrt((t_sq - self.k) / norm_sq.clamp_min(1e-8))
        x_proj = x_narrow * scale
        
        # Ensure time component is positive
        t_proj = torch.sqrt(self.k + torch.sum(x_proj * x_proj, dim=-1, keepdim=True))
        
        return torch.cat([t_proj, x_proj], dim=-1)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from manifold to tangent space at origin."""
        # x: [batch, d+1] where d = feature_dim
        d = x.size(-1) - 1
        t = x.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, d)
        
        # Compute arccosh(t / sqrt(k))
        k_sqrt = torch.sqrt(self.k)
        arg = (t / k_sqrt).clamp(min=1.0 + 1e-8)
        theta = torch.acosh(arg)
        
        # Compute norm of spatial part
        norm = torch.norm(x_narrow, dim=-1, keepdim=True).clamp_min(1e-8)
        
        # log map: (theta / ||x||) * x
        result = (theta / norm) * x_narrow
        return result

    def inner(self, x: torch.Tensor, y: torch.Tensor, keepdim: bool = False) -> torch.Tensor:
        """Minkowski inner product: x[0]*y[0] - x[1:]*y[1:]."""
        d = x.size(-1) - 1
        t_x = x.narrow(-1, 0, 1)
        t_y = y.narrow(-1, 0, 1)
        x_narrow = x.narrow(-1, 1, d)
        y_narrow = y.narrow(-1, 1, d)
        
        time_part = t_x * t_y
        space_part = torch.sum(x_narrow * y_narrow, dim=-1, keepdim=True)
        
        result = time_part - space_part
        if not keepdim:
            result = result.squeeze(-1)
        return result
