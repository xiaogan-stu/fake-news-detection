# -*- coding: utf-8 -*-
"""
baselines.py
------------
假新闻检测基线模型集合，供 scripts/run_experiment.py 调用。

所有模型统一接口：
    __init__(self, in_dim, gnn_dim, num_layers, dropout=0.1)
    forward(self, x, edge_index, batch) → logits [batch_size, 2]

其中：
    x:          [num_nodes, in_dim]   节点特征
    edge_index: [2, num_edges]        COO 边索引
    batch:      [num_nodes]           节点所属子图索引（PyG Batch 格式）

分类基于每棵传播树的根节点（batch 中每图的第 0 号本地节点）。

包含模型：
    1. MLPDetector      — 纯文本 MLP，无图结构
    2. GCNDetector       — 图卷积网络
    3. GATDetector       — 图注意力网络
    4. SAGEDetector      — GraphSAGE
    5. BiGCNDetector     — 双向 GCN（谣言检测专用）
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


def _root_node_indices(batch: torch.Tensor) -> torch.Tensor:
    """
    提取 batch 中每个子图根节点的全局索引。

    PyG Batch 拼接顺序中，每个子图的第一个节点即为本地索引 0（新闻根节点）。
    识别方法：batch 向量首次变化的位置（含第 0 个节点）。

    参数：
        batch: [num_nodes]，值范围 0..B-1，同一子图节点连续排列

    返回：
        [batch_size] 每个子图根节点的全局索引
    """
    if batch.numel() == 0:
        return batch.new_zeros((0,), dtype=torch.long)
    change = torch.ones(batch.numel(), dtype=torch.bool, device=batch.device)
    change[1:] = batch[1:] != batch[:-1]
    return torch.nonzero(change, as_tuple=True)[0]


# ═══════════════════════════════════════════════════════════════════
# 1. MLPDetector — 纯文本基线，不使用图结构
# ═══════════════════════════════════════════════════════════════════

class MLPDetector(nn.Module):
    """
    纯 MLP 基线：不使用 edge_index，仅对根节点特征做多层感知机分类。
    作为"无图结构"的消融对照。

    结构：Linear(in_dim → 256) → ReLU → Dropout → Linear(256 → gnn_dim) → ReLU → Dropout → Linear(gnn_dim → 2)
    """

    def __init__(self, in_dim: int, gnn_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        """
        参数：
            in_dim:     输入节点特征维度（如 4096）
            gnn_dim:    隐藏层维度
            num_layers: 未使用（保持接口统一，MLP 固定两层隐藏层）
            dropout:    Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.dropout = dropout

        # MLP: in_dim → 256 → gnn_dim → 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),       # [num_nodes, in_dim] → [num_nodes, 256]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, gnn_dim),      # [num_nodes, 256] → [num_nodes, gnn_dim]
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gnn_dim, 2),        # [num_nodes, gnn_dim] → [num_nodes, 2]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播（忽略 edge_index）。

        参数：
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]（未使用）
            batch:      [num_nodes]

        返回：
            [batch_size, 2] 根节点的分类 logits
        """
        # 全部节点过 MLP: [num_nodes, in_dim] → [num_nodes, 2]
        logits_all = self.mlp(x)

        # 提取根节点: [batch_size, 2]
        root_ix = _root_node_indices(batch)
        return logits_all.index_select(0, root_ix)


# ═══════════════════════════════════════════════════════════════════
# 2. GCNDetector — 图卷积网络
# ═══════════════════════════════════════════════════════════════════

class GCNDetector(nn.Module):
    """
    基于 GCNConv 的假新闻检测器。

    结构：
        Linear(in_dim → gnn_dim) → num_layers × [GCNConv → ReLU → Dropout] → 根节点分类
    """

    def __init__(self, in_dim: int, gnn_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        """
        参数：
            in_dim:     输入节点特征维度
            gnn_dim:    GNN 隐藏维度
            num_layers: GCNConv 层数
            dropout:    Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        self.input_proj = nn.Linear(in_dim, gnn_dim)

        # num_layers 层 GCNConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(gnn_dim, gnn_dim))

        # 分类头: [batch_size, gnn_dim] → [batch_size, 2]
        self.classifier = nn.Linear(gnn_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]
            batch:      [num_nodes]

        返回：
            [batch_size, 2]
        """
        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        h = self.input_proj(x)
        h = h.relu()

        # 多层 GCN: [num_nodes, gnn_dim] → [num_nodes, gnn_dim]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = h.relu()
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        # 提取根节点: [batch_size, gnn_dim]
        root_ix = _root_node_indices(batch)
        h_root = h.index_select(0, root_ix)

        # 分类: [batch_size, gnn_dim] → [batch_size, 2]
        return self.classifier(h_root)


# ═══════════════════════════════════════════════════════════════════
# 3. GATDetector — 图注意力网络
# ═══════════════════════════════════════════════════════════════════

class GATDetector(nn.Module):
    """
    基于 GATConv 的假新闻检测器。

    结构：
        Linear(in_dim → gnn_dim) → num_layers × [GATConv(heads=4, concat=False) → ReLU → Dropout] → 根节点分类

    注意：GATConv(heads=4, concat=False) 输出维度仍为 gnn_dim，
    即每个头输出 gnn_dim//heads 维，4 头拼接后再平均回 gnn_dim。
    """

    def __init__(self, in_dim: int, gnn_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        """
        参数：
            in_dim:     输入节点特征维度
            gnn_dim:    GNN 隐藏维度（每层输出维度）
            num_layers: GATConv 层数
            dropout:    Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        self.input_proj = nn.Linear(in_dim, gnn_dim)

        # num_layers 层 GATConv (heads=4, concat=False → 输出维度 = gnn_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(gnn_dim, gnn_dim, heads=4, concat=False, dropout=dropout))

        # 分类头
        self.classifier = nn.Linear(gnn_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]
            batch:      [num_nodes]

        返回：
            [batch_size, 2]
        """
        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        h = self.input_proj(x)
        h = h.relu()

        # 多层 GAT: [num_nodes, gnn_dim] → [num_nodes, gnn_dim]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = h.relu()
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        # 提取根节点: [batch_size, gnn_dim]
        root_ix = _root_node_indices(batch)
        h_root = h.index_select(0, root_ix)

        # 分类: [batch_size, gnn_dim] → [batch_size, 2]
        return self.classifier(h_root)


# ═══════════════════════════════════════════════════════════════════
# 4. SAGEDetector — GraphSAGE
# ═══════════════════════════════════════════════════════════════════

class SAGEDetector(nn.Module):
    """
    基于 SAGEConv 的假新闻检测器。

    结构：
        Linear(in_dim → gnn_dim) → num_layers × [SAGEConv → ReLU → Dropout] → 根节点分类
    """

    def __init__(self, in_dim: int, gnn_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        """
        参数：
            in_dim:     输入节点特征维度
            gnn_dim:    GNN 隐藏维度
            num_layers: SAGEConv 层数
            dropout:    Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        self.input_proj = nn.Linear(in_dim, gnn_dim)

        # num_layers 层 SAGEConv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(gnn_dim, gnn_dim))

        # 分类头
        self.classifier = nn.Linear(gnn_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]
            batch:      [num_nodes]

        返回：
            [batch_size, 2]
        """
        # 输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        h = self.input_proj(x)
        h = h.relu()

        # 多层 SAGE: [num_nodes, gnn_dim] → [num_nodes, gnn_dim]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = h.relu()
            h = nn.functional.dropout(h, p=self.dropout, training=self.training)

        # 提取根节点: [batch_size, gnn_dim]
        root_ix = _root_node_indices(batch)
        h_root = h.index_select(0, root_ix)

        # 分类: [batch_size, gnn_dim] → [batch_size, 2]
        return self.classifier(h_root)


# ═══════════════════════════════════════════════════════════════════
# 5. BiGCNDetector — 双向 GCN（谣言检测专用）
# ═══════════════════════════════════════════════════════════════════

class BiGCNDetector(nn.Module):
    """
    双向 GCN 假新闻检测器，参考 BiGCN 论文思路。

    核心思想：谣言传播树中，信息既有自顶向下（根→叶）的传播，
    也有自底向上（叶→根）的反馈。两条路径的语义不同，应分别建模。

    实现：
        - 自顶向下（TD）：使用原始 edge_index，消息从父节点传向子节点
        - 自底向上（BU）：使用 edge_index 的转置（flip），消息从子节点传向父节点
        - 两个方向各 num_layers 层 GCNConv
        - 最后拼接两个方向的根节点特征 [gnn_dim * 2] → 分类

    结构：
        Linear(in_dim → gnn_dim) × 2（TD / BU 各一个投影）
        → num_layers × [GCNConv(TD) + GCNConv(BU)] → ReLU → Dropout
        → 拼接根节点特征 → Linear(gnn_dim*2 → 2)
    """

    def __init__(self, in_dim: int, gnn_dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        """
        参数：
            in_dim:     输入节点特征维度
            gnn_dim:    GNN 隐藏维度（每个方向）
            num_layers: 每个方向的 GCNConv 层数
            dropout:    Dropout 概率
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # 两个方向各自的输入投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        self.td_proj = nn.Linear(in_dim, gnn_dim)
        self.bu_proj = nn.Linear(in_dim, gnn_dim)

        # 自顶向下 GCN 层
        self.td_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.td_convs.append(GCNConv(gnn_dim, gnn_dim))

        # 自底向上 GCN 层
        self.bu_convs = nn.ModuleList()
        for _ in range(num_layers):
            self.bu_convs.append(GCNConv(gnn_dim, gnn_dim))

        # 分类头: 拼接后 [batch_size, gnn_dim * 2] → [batch_size, 2]
        self.classifier = nn.Linear(gnn_dim * 2, 2)

    @staticmethod
    def _flip_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
        """
        翻转边方向：将 (src → dst) 变为 (dst → src)。

        参数：
            edge_index: [2, num_edges]，第 0 行为源，第 1 行为目标

        返回：
            [2, num_edges]，源与目标互换
        """
        return edge_index.flip(0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            x:          [num_nodes, in_dim]
            edge_index: [2, num_edges]
            batch:      [num_nodes]

        返回：
            [batch_size, 2]
        """
        # 翻转边索引，构建自底向上传播路径
        edge_index_bu = self._flip_edge_index(edge_index)

        # ---- 自顶向下分支 ----
        # 投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        h_td = self.td_proj(x).relu()
        # 多层 GCN: [num_nodes, gnn_dim] → [num_nodes, gnn_dim]
        for conv in self.td_convs:
            h_td = conv(h_td, edge_index)
            h_td = h_td.relu()
            h_td = nn.functional.dropout(h_td, p=self.dropout, training=self.training)

        # ---- 自底向上分支 ----
        # 投影: [num_nodes, in_dim] → [num_nodes, gnn_dim]
        h_bu = self.bu_proj(x).relu()
        # 多层 GCN: [num_nodes, gnn_dim] → [num_nodes, gnn_dim]
        for conv in self.bu_convs:
            h_bu = conv(h_bu, edge_index_bu)
            h_bu = h_bu.relu()
            h_bu = nn.functional.dropout(h_bu, p=self.dropout, training=self.training)

        # 提取根节点
        root_ix = _root_node_indices(batch)
        h_td_root = h_td.index_select(0, root_ix)   # [batch_size, gnn_dim]
        h_bu_root = h_bu.index_select(0, root_ix)   # [batch_size, gnn_dim]

        # 拼接两个方向的根节点特征: [batch_size, gnn_dim * 2]
        h_root = torch.cat([h_td_root, h_bu_root], dim=-1)

        # 分类: [batch_size, gnn_dim * 2] → [batch_size, 2]
        return self.classifier(h_root)
