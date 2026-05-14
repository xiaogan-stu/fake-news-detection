# -*- coding: utf-8 -*-
"""
HyperbolicFakeNewsDetector
---------------------------
基于洛伦兹流形的双曲假新闻检测模型。

在训练过程中不加载 Llama：仅消费预提取的节点特征 x (4096维)，
通过多层双曲图卷积在洛伦兹流形上进行传播，
然后映射回欧氏空间进行二分类。

关键步骤的张量形状标注为 ``# [..., ...]``。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.layers.hgnn_layer import HyperbolicGraphConvolution

from src.models.manifolds.lorentz import Lorentz as LorentzManifold


def _add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """为边索引添加自环，稳定邻域聚合。"""
    device = edge_index.device
    sl = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, sl], dim=1)


def _edge_index_to_sparse_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    将 PyG 的 edge_index 转换为稀疏邻接矩阵。
    
    PyG 约定：edge_index[0] 是源节点，edge_index[1] 是目标节点；
    使用 adj[target, source] = 1 进行聚合，将邻居信息收集到目标行。
    """
    ei = _add_self_loops(edge_index, num_nodes)
    row = ei[1]  # target
    col = ei[0]  # source
    values = torch.ones(row.numel(), device=ei.device, dtype=torch.float32)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]),
        values,
        (num_nodes, num_nodes),
        device=ei.device,
        dtype=torch.float32,
    ).coalesce()
    return adj


class HyperbolicFakeNewsDetector(nn.Module):
    """
    基于洛伦兹流形的双曲假新闻检测器。
    
    仅输出每个传播树根节点的二分类 logits，与图级标签对齐。
    """

    def __init__(
        self,
        in_dim: int = 4096,
        gnn_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        curvature: float = 1.0,
        hidden_mlp: int = 512,
    ) -> None:
        """
        初始化双曲假新闻检测器。
        
        参数：
            in_dim: 输入节点特征维度（Llama隐藏维度，默认4096）
            gnn_dim: 投影后的欧氏特征维度，也是GNN输入/输出维度（默认128）
            num_layers: 双曲图卷积层数（默认2）
            dropout: MLP和HGNN层的dropout概率（默认0.1）
            curvature: 洛伦兹流形曲率（默认1.0，负值表示双曲空间）
            hidden_mlp: MLP第一层隐藏层宽度（默认512）
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.curvature = curvature

        # MLP降维: in_dim -> hidden_mlp -> gnn_dim
        # 每一层后添加 LayerNorm + ReLU + Dropout
        self.feat_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_mlp),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_mlp),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, gnn_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(gnn_dim),
            nn.Dropout(dropout),
        )

        # 洛伦兹流形，固定曲率
        # curvature 为负数表示双曲空间（通常使用 -1.0）
        self.manifold = LorentzManifold(k=abs(curvature), learnable=False)

        # 构建多层双曲图卷积
        # 注意：HGNN 期望输入为流形坐标（时间+空间），所以维度是 gnn_dim + 1
        self.hgnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = HyperbolicGraphConvolution(
                manifold=self.manifold,
                in_features=gnn_dim + 1,
                out_features=gnn_dim + 1,
                use_bias=True,
                dropout=dropout,
                use_att=False,
                local_agg=True,
            )
            self.hgnn_layers.append(layer)

        # 分类器：将根节点特征映射到二分类logits
        self.classifier = nn.Linear(gnn_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数：
            x: [num_nodes, in_dim] 预提取的Llama节点特征（4096维）
            edge_index: [2, num_edges] PyG COO格式边索引（第0行为源节点）
            batch: [num_nodes] 子图ID，同一图的节点必须连续
        
        返回：
            [batch_size, 2] 仅根节点的二分类logits
        """
        # --- 步骤1: 欧氏MLP降维: [num_nodes, 4096] -> [num_nodes, 512] -> [num_nodes, 128] ---
        z = self.feat_proj(x)
        # z: [num_nodes, gnn_dim]

        # --- 步骤2: 添加时间维度（初始化为0）: [num_nodes, 128] -> [num_nodes, 129] ---
        # MHR 的流形方法期望输入已经包含时间维度（第一列）
        z_with_time = torch.cat([torch.zeros(z.size(0), 1, device=z.device), z], dim=-1)
        # z_with_time: [num_nodes, gnn_dim + 1]

        # --- 步骤3: 投影到原点切空间: [num_nodes, 129] -> [num_nodes, 129] ---
        u = self.manifold.proju0(z_with_time)
        # u: [num_nodes, gnn_dim + 1]

        # --- 步骤4: 指数映射到洛伦兹流形: [num_nodes, 129] -> [num_nodes, 129] ---
        h = self.manifold.expmap0(u, project=True)
        # h: [num_nodes, gnn_dim + 1] (时间维度 + 空间维度)

        # --- 步骤5: 构建稀疏邻接矩阵 ---
        num_nodes = x.size(0)
        adj = _edge_index_to_sparse_adj(edge_index, num_nodes)

        # --- 步骤6: 多层双曲图卷积 ---
        for i, layer in enumerate(self.hgnn_layers):
            h, _ = layer((h, adj))
            # h: [num_nodes, gnn_dim + 1]
            
            # 每层后调用流形投影校正，防止特征脱离流形
            h = self.manifold.projx(h)

        # --- 步骤7: 对数映射回到切空间: [num_nodes, 129] -> [num_nodes, 128] ---
        t = self.manifold.logmap0(h)
        # t: [num_nodes, gnn_dim]（MHR 的 logmap0 会自动去掉时间维度）

        # --- 步骤8: 提取根节点特征 ---
        root_ix = self._root_node_indices(batch)
        # root_ix: [batch_size]
        h_root = t.index_select(0, root_ix)
        # h_root: [batch_size, gnn_dim]

        # --- 步骤9: 分类器输出 ---
        logits = self.classifier(h_root)
        # logits: [batch_size, 2]

        return logits

    @staticmethod
    def _root_node_indices(batch: torch.Tensor) -> torch.Tensor:
        """
        在标准PyG Batch拼接顺序中，每个子图的第一个节点是本地索引0（新闻根节点）。
        
        识别方法：batch向量首次变化的位置（包括第一个节点）。
        
        参数：
            batch: [num_nodes]，值范围0..B-1，同一图的节点必须连续
        
        返回：
            [batch_size] 每个子图根节点的全局索引
        """
        if batch.numel() == 0:
            return batch.new_zeros((0,), dtype=torch.long)
        change = torch.ones(batch.numel(), dtype=torch.bool, device=batch.device)
        change[1:] = batch[1:] != batch[:-1]
        roots = torch.nonzero(change, as_tuple=True)[0]
        return roots

    def __repr__(self) -> str:
        return (
            f"HyperbolicFakeNewsDetector(in_dim={self.in_dim}, gnn_dim={self.gnn_dim}, "
            f"num_layers={self.num_layers}, dropout={self.dropout}, curvature={self.curvature})"
        )


# 测试代码
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试用户要求的接口
    print("=" * 60)
    print("测试: 用户要求的接口")
    print("=" * 60)
    
    model = HyperbolicFakeNewsDetector(
        in_dim=4096, 
        gnn_dim=128, 
        num_layers=2, 
        dropout=0.1, 
        curvature=-1.0
    ).to(device)
    
    print(f"模型: {model}")
    
    # 测试输入
    x = torch.randn(10, 4096, device=device)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8],[1,2,3,4,0,6,7,8,5]], dtype=torch.long, device=device)
    batch = torch.tensor([0,0,0,0,0,1,1,1,1,1], device=device)
    
    print(f"\n输入 x shape: {x.shape}")
    print(f"输入 edge_index shape: {edge_index.shape}")
    print(f"输入 batch shape: {batch.shape}")
    
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index, batch)
    
    print(f"\n输出 logits shape: {logits.shape}")
    print(f"期望输出 shape: (2, 2)")
    
    assert logits.shape == (2, 2), f"输出形状错误: {logits.shape}"
    assert not torch.isnan(logits).any(), "输出包含NaN"
    
    print("\n✅ 验证通过！")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数数: {total_params:,}")
