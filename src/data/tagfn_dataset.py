"""
TAGFN 数据集加载器
数据来源: kayzliu/TAGFN HuggingFace 仓库

支持两个子集:
  - Politifact: 314 图, 41054 节点, 40740 边
  - Gossipcop:  5464 图, 314262 节点, 308798 边

两个子集的文件结构完全相同:
  A.txt              - 全局边列表 "src, dst"
  graph_labels.npy   - [N] 图标签 (0=真, 1=假)
  node_graph_id.npy  - [M] 每节点所属图ID
  node_time.npy      - [M] 节点时间戳 (根节点=0)
  train/val/test_idx.npy - 图级别划分索引
  raw_text/part-00000.parquet - [M, 1] 节点文本, 列名'0'
"""

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import List, Optional


class FakeNewsDataset:
    """
    假新闻数据集公共基类。

    封装了 TAGFN 格式数据集的通用加载逻辑：
    边列表、图标签、节点元数据、文本、预提取特征、图索引构建、PyG Data 生成。

    子类只需指定默认数据目录路径即可。
    """

    # 子类可覆盖：默认数据目录
    DEFAULT_DATA_DIR: str = ""

    def __init__(self, data_dir: Optional[str] = None, feature_cache_path: Optional[str] = None):
        """
        参数：
            data_dir:           数据集根目录，如 /mnt/workspace/data/politifact
                                若为 None 则使用子类的 DEFAULT_DATA_DIR
            feature_cache_path: Llama-3 预提取特征路径 (.pt)，训练阶段传入
        """
        self.data_dir = data_dir or self.DEFAULT_DATA_DIR
        self.feature_cache_path = feature_cache_path
        self._load_raw_data()
        self._build_graph_edge_index()

    def _load_raw_data(self):
        """加载原始数据文件：边列表、图标签、节点元数据、文本、预提取特征"""
        print(f"[数据] 加载 {self.data_dir} 下的图结构文件...")

        # ---- 边列表: "src, dst" 格式, 全局节点索引 ----
        edges = []
        with open(os.path.join(self.data_dir, 'A.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    edges.append((int(parts[0].strip()), int(parts[1].strip())))
        self.global_edges = np.array(edges, dtype=np.int64)

        # ---- 图/节点元数据 ----
        self.graph_labels   = np.load(os.path.join(self.data_dir, 'graph_labels.npy'))
        self.node_graph_id  = np.load(os.path.join(self.data_dir, 'node_graph_id.npy'))
        self.node_time      = np.load(os.path.join(self.data_dir, 'node_time.npy'))
        self.train_idx      = np.load(os.path.join(self.data_dir, 'train_idx.npy'))
        self.val_idx        = np.load(os.path.join(self.data_dir, 'val_idx.npy'))
        self.test_idx       = np.load(os.path.join(self.data_dir, 'test_idx.npy'))

        # ---- 节点文本 ----
        print("[数据] 加载节点文本...")
        df = pd.read_parquet(
            os.path.join(self.data_dir, 'raw_text', 'part-00000.parquet')
        )
        self.node_texts = df['0'].fillna('').tolist()

        # ---- 预提取特征（可选）----
        self.node_features = None
        if self.feature_cache_path and os.path.exists(self.feature_cache_path):
            print(f"[数据] 加载预提取特征: {self.feature_cache_path}")
            self.node_features = torch.load(self.feature_cache_path, map_location='cpu')
            print(f"  特征形状: {self.node_features.shape}")

        print(f"[数据] 加载完成: {len(self.graph_labels)} 图, "
              f"{len(self.node_texts)} 节点, {len(self.global_edges)} 边")

    def _build_graph_edge_index(self):
        """O(E) 预处理: 按图分组节点和边，加速后续单图构建"""
        print("[数据] 建立图索引...")
        num_graphs = len(self.graph_labels)

        # 每个图的全局节点列表
        self.graph_to_nodes = [[] for _ in range(num_graphs)]
        for node_id, g_id in enumerate(self.node_graph_id):
            self.graph_to_nodes[g_id].append(node_id)
        self.graph_to_nodes = [np.array(v, dtype=np.int64)
                               for v in self.graph_to_nodes]

        # 每个图的边列表（全局索引）
        self.graph_to_edges = [[] for _ in range(num_graphs)]
        for src, dst in self.global_edges:
            g_id = int(self.node_graph_id[src])
            self.graph_to_edges[g_id].append((src, dst))

        print("[数据] 图索引建立完成")

    def get_graph_data(self, graph_idx: int) -> Data:
        """
        构建单个图的 PyG Data 对象

        返回：
            data.x:               [num_nodes, feat_dim] 或 None
            data.edge_index:      [2, num_edges]
            data.y:               [1] 标签
            data.num_nodes:       节点数
            data.root_idx:        根节点的本地索引
            data.global_node_ids: [num_nodes] 全局节点ID（供特征切片用）
            data.node_texts:      List[str] 节点文本
        """
        global_node_ids = self.graph_to_nodes[graph_idx]
        n = len(global_node_ids)

        # 全局 → 本地索引映射
        global_to_local = {int(gid): lid for lid, gid in enumerate(global_node_ids)}

        # 构建本地 edge_index
        raw_edges = self.graph_to_edges[graph_idx]
        if len(raw_edges) > 0:
            local_edges = [[global_to_local[s], global_to_local[d]]
                           for s, d in raw_edges]
            edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
            # [2, num_edges]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 根节点 = 时间戳最小的节点（通常为 0）
        times = self.node_time[global_node_ids]
        root_local_idx = int(np.argmin(times))

        # 标签
        y = torch.tensor([int(self.graph_labels[graph_idx])], dtype=torch.long)

        # 节点文本
        texts = [self.node_texts[i] for i in global_node_ids]

        # 节点特征（若已提取）
        if self.node_features is not None:
            x = self.node_features[global_node_ids]  # [n, 4096]
        else:
            x = None

        return Data(
            x=x,
            edge_index=edge_index,
            y=y,
            num_nodes=n,
            root_idx=root_local_idx,
            global_node_ids=torch.tensor(global_node_ids, dtype=torch.long),
            node_texts=texts,
        )

    def get_split(self, split: str) -> List[Data]:
        """获取 train / val / test 划分"""
        assert split in ('train', 'val', 'test'), f"split 必须是 train/val/test"
        idx_map = {'train': self.train_idx,
                   'val':   self.val_idx,
                   'test':  self.test_idx}
        data_list = [self.get_graph_data(int(i)) for i in idx_map[split]]
        print(f"[数据] {split}: {len(data_list)} 个图")
        return data_list

    def get_all_node_texts(self) -> List[str]:
        """返回所有节点的文本（供 LLM 特征提取用）"""
        return self.node_texts

    @property
    def num_graphs(self):
        """图数量"""
        return len(self.graph_labels)

    @property
    def num_nodes(self):
        """节点总数"""
        return len(self.node_texts)


class PolitifactDataset(FakeNewsDataset):
    """
    Politifact 数据集（314 图, 41054 节点, 40740 边）。

    数据来源: kayzliu/TAGFN HuggingFace 仓库的 politifact 子集。
    """

    DEFAULT_DATA_DIR = "/mnt/workspace/data/politifact"


class GossipcopDataset(FakeNewsDataset):
    """
    Gossipcop 数据集（5464 图, 314262 节点, 308798 边）。

    数据来源: kayzliu/TAGFN HuggingFace 仓库的 gossipcop 子集。
    文件结构与 Politifact 完全相同，仅数据规模不同。
    """

    DEFAULT_DATA_DIR = "/mnt/workspace/data/gossipcop"


if __name__ == '__main__':
    import sys

    # 默认测试 Politifact
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "politifact"
    data_dir_map = {
        "politifact": "/mnt/workspace/data/politifact",
        "gossipcop":  "/mnt/workspace/data/gossipcop",
    }
    cls_map = {
        "politifact": PolitifactDataset,
        "gossipcop":  GossipcopDataset,
    }

    if dataset_name not in cls_map:
        print(f"未知数据集: {dataset_name}，可选: {list(cls_map.keys())}")
        sys.exit(1)

    data_dir = data_dir_map[dataset_name]
    if not os.path.isdir(data_dir):
        # Windows 本地测试时可能没有数据，给出提示
        print(f"[提示] 数据目录不存在: {data_dir}")
        print(f"  请先下载 {dataset_name} 数据集到该路径")
        sys.exit(0)

    print(f"{'=' * 60}")
    print(f"测试数据集: {dataset_name}")
    print(f"{'=' * 60}")

    ds = cls_map[dataset_name](data_dir=data_dir)

    # 图数量 & 标签分布
    labels = ds.graph_labels
    fake_count = int((labels == 1).sum())
    real_count = int((labels == 0).sum())
    print(f"\n图数量: {ds.num_graphs} (fake={fake_count}, real={real_count})")
    print(f"节点总数: {ds.num_nodes}")
    print(f"边总数: {len(ds.global_edges)}")

    # 划分统计
    print(f"划分: train={len(ds.train_idx)}, val={len(ds.val_idx)}, test={len(ds.test_idx)}")

    # 验证单图构建
    sample = ds.get_graph_data(0)
    print(f"\n示例图[0]:")
    print(f"  num_nodes  = {sample.num_nodes}")
    print(f"  edge_index = {sample.edge_index.shape}")
    print(f"  y          = {sample.y.item()}")
    print(f"  root_idx   = {sample.root_idx}")
    print(f"  文本数量   = {len(sample.node_texts)}")

    # 验证划分
    train_data = ds.get_split('train')
    val_data   = ds.get_split('val')
    test_data  = ds.get_split('test')
    print(f"\n划分验证: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print(f"\n{dataset_name} 数据集测试通过!")
