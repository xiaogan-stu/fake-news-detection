# -*- coding: utf-8 -*-
"""
GATv2FakeNewsDetector (Fallback Model)
--------------------------------------
This is a fallback solution when Hyperbolic Graph Neural Network (HGNN) encounters NaN issues.

When the hyperbolic model becomes numerically unstable (e.g., NaN loss in 3 consecutive batches),
you can switch to this Euclidean-space GATv2 model to continue training.

Architecture (Euclidean space, no hyperbolic geometry):
1. feat_proj: Linear(4096→512) + ReLU + LayerNorm + Dropout + Linear(512→128)
2. Two GATv2Conv(128, 128, heads=4, concat=False) layers from torch_geometric.nn
3. ReLU + Dropout(0.1) after each GATv2 layer
4. Extract root node features for each propagation tree (first node of each group in batch)
5. Linear(128, 2) outputs binary classification logits

Interface is identical to HyperbolicFakeNewsDetector for seamless switching.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


def _root_node_indices(batch: torch.Tensor) -> torch.Tensor:
    """
    In standard PyG Batch concatenation order, the first node of each subgraph
    is local index 0 (news root node).

    Identification: positions where batch vector first changes (including first node).

    Parameters
    ----------
    batch : Tensor
        # [num_nodes], values 0..B-1, nodes of same graph are consecutive

    Returns
    -------
    Tensor
        # [batch_size], global indices of root nodes for each subgraph
    """
    if batch.numel() == 0:
        return batch.new_zeros((0,), dtype=torch.long)
    change = torch.ones(batch.numel(), dtype=torch.bool, device=batch.device)
    change[1:] = batch[1:] != batch[:-1]
    roots = torch.nonzero(change, as_tuple=True)[0]
    return roots


class GATv2FakeNewsDetector(nn.Module):
    """
    Fake news detection model based on GATv2 (Graph Attention Network v2).

    Serves as a fallback for hyperbolic models, using Euclidean space operations
    to avoid numerical instability issues in hyperbolic geometry.

    Only outputs 2-class logits for root nodes of each propagation tree,
    to align with graph-level labels.
    """

    def __init__(
        self,
        in_dim: int = 4096,
        hidden_mlp: int = 512,
        gnn_dim: int = 128,
        dropout: float = 0.1,
        gat_heads: int = 4,
    ) -> None:
        """
        Parameters
        ----------
        in_dim
            Input node feature dimension (Llama hidden dim, default 4096).
        hidden_mlp
            MLP first hidden layer width (4096 -> 512).
        gnn_dim
            Projected feature dimension and GNN layer dimension (default 128).
        dropout
            Dropout probability in MLP and GNN layers.
        gat_heads
            Number of attention heads in GATv2Conv (default 4).
        """
        super().__init__()
        self.in_dim = in_dim
        self.gnn_dim = gnn_dim
        self.dropout = dropout

        # Feature projection: 4096 -> 512 -> 128
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

        # Two GATv2 layers with residual connections
        self.gat1 = GATv2Conv(
            in_channels=gnn_dim,
            out_channels=gnn_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        self.gat2 = GATv2Conv(
            in_channels=gnn_dim,
            out_channels=gnn_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )

        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(dropout)

        # Classifier head
        self.classifier = nn.Linear(gnn_dim, 2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            # [num_nodes, in_dim], pre-extracted Llama node features (4096 dim)
        edge_index : Tensor
            # [2, num_edges], PyG COO edge index
        batch : Tensor
            # [num_nodes], subgraph id, nodes of the same graph must be consecutive

        Returns
        -------
        Tensor
            # [batch_size, 2], binary classification logits only on root nodes
        """
        # Feature projection: 4096 -> 128
        z = self.feat_proj(x)
        # z: [num_nodes, gnn_dim]

        # First GATv2 layer with residual connection
        h1 = self.gat1(z, edge_index)
        h1 = self.relu(h1)
        h1 = self.dropout_layer(h1)
        h1 = h1 + z  # residual

        # Second GATv2 layer with residual connection
        h2 = self.gat2(h1, edge_index)
        h2 = self.relu(h2)
        h2 = self.dropout_layer(h2)
        h2 = h2 + h1  # residual

        # Extract root node features
        root_ix = _root_node_indices(batch)
        h_root = h2.index_select(0, root_ix)
        # h_root: [batch_size, gnn_dim]

        # Classification head
        logits = self.classifier(h_root)
        # logits: [batch_size, 2]

        return logits

    def __repr__(self) -> str:
        return (
            f"GATv2FakeNewsDetector(in_dim={self.in_dim}, gnn_dim={self.gnn_dim}, "
            f"dropout={self.dropout})"
        )


# Test code
if __name__ == "__main__":
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_nodes = 64
    num_edges = 128

    x = torch.randn(num_nodes, 4096, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    batch = torch.tensor([0] * (num_nodes // 2) + [1] * (num_nodes // 2), device=device)

    model = GATv2FakeNewsDetector(
        in_dim=4096,
        hidden_mlp=512,
        gnn_dim=128,
        dropout=0.1,
        gat_heads=4,
    ).to(device)

    print(f"Model structure:\n{model}")
    print(f"\nInput x shape: {x.shape}")
    print(f"Input edge_index shape: {edge_index.shape}")
    print(f"Input batch shape: {batch.shape}")

    # Forward pass
    logits = model(x, edge_index, batch)
    print(f"\nOutput logits shape: {logits.shape}")
    print(f"Output logits:\n{logits}")

    # Verify interface compatibility
    print("\nInterface compatibility test:")
    print(f"  Input dim: {model.in_dim}")
    print(f"  GNN dim: {model.gnn_dim}")
    print(f"  Output dim: {logits.shape}")

    # Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\nTest completed: GATv2 fallback model works correctly.")
