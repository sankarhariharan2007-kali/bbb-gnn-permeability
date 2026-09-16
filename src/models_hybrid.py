"""
Priority 1: hybrid model -- graph embedding + descriptors, concatenated
before the classifier head.

Additive file, same convention as models_gine.py: the base four-operator
comparison in models.py/train.py/run_all.py is untouched. This duplicates the
GNN skeleton (depth, hidden width, norm, readout) on purpose so a hybrid-vs-
base comparison isolates the effect of adding descriptors, the same way the
original file isolates the effect of the conv operator.

Only the classifier head changes: the pooled graph representation
[mean_pool || max_pool] is concatenated with an 8-dim standardized descriptor
vector (src/hybrid_features.py) before the two-layer head, instead of going
into it directly. Everything upstream of the head -- the four conv operators,
BatchNorm, dropout -- is byte-for-byte identical to models.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .featurize import NODE_DIM
from .hybrid_features import DESCRIPTOR_DIM
from .models import MODELS, _make_conv
from torch_geometric.nn import global_max_pool, global_mean_pool


class HybridGNNClassifier(nn.Module):
    def __init__(
        self,
        conv: str,
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        descriptor_dim: int = DESCRIPTOR_DIM,
    ) -> None:
        super().__init__()
        if conv not in MODELS:
            raise ValueError(f"unknown conv type: {conv!r} (expected one of {MODELS})")
        self.conv_type = conv
        self.descriptor_dim = descriptor_dim
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                _make_conv(conv, in_dim if layer == 0 else hidden, hidden, heads)
            )
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)

        # Only this changes relative to GNNClassifier: descriptor_dim extra
        # input features into the first head layer.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + descriptor_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        graph_repr = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        # data.descriptors is attached per-graph as shape (1, D) by
        # train_hybrid.py before batching; PyG's default collate concatenates
        # per-graph tensor attributes along dim 0, same as it does for `y`,
        # so this arrives here as (batch_size, D).
        combined = torch.cat([graph_repr, data.descriptors], dim=1)
        return self.head(combined).squeeze(-1)


def build_hybrid_model(conv: str, **kwargs) -> HybridGNNClassifier:
    return HybridGNNClassifier(conv=conv, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
