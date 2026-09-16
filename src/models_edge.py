"""
Priority 2: give edge features to all three operators that can use them, not
just GIN.

Same skeleton convention as models_gine.py (additive, doesn't touch
models.py): depth, hidden width, norm, readout, head, and training loop are
identical across gine / gat_edge / sage_edge, so a gap between them and their
edge-blind counterparts (gin / gat / sage in models.py) is attributable to
the edge signal, not incidental capacity differences.

- gine: GINEConv, unchanged from models_gine.py (kept here too so all three
  edge-aware operators live in one place).
- gat_edge: PyG's GATConv natively supports `edge_dim`, projecting edge_attr
  into the attention computation -- no custom operator needed, so this reuses
  the well-tested built-in rather than reimplementing GAT's attention math.
- sage_edge: plain SAGEConv has no edge_attr hook, so this adds a small
  custom MessagePassing operator, SAGEEdgeConv, using the same edge-injection
  convention GINEConv already uses (add the projected edge vector into the
  neighbor feature before aggregating) but with SAGE's own combination rule
  (separate self/neighbor linear transforms, mean aggregation) rather than
  GIN's sum-then-MLP. This is the standard "edge-conditioned SAGE" pattern
  referred to in the priority list; it is not a PyG built-in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINEConv, MessagePassing
from torch_geometric.nn import global_max_pool, global_mean_pool

from .featurize import EDGE_DIM, NODE_DIM

EDGE_MODELS = ["gine", "gat_edge", "sage_edge"]


class SAGEEdgeConv(MessagePassing):
    """Edge-conditioned GraphSAGE: out = lin_l(x) + lin_r(mean_j(x_j + edge_ij)).

    Mirrors PyG's own SAGEConv formula (separate self/neighbor transforms,
    mean aggregation) with one change: the edge feature, linearly projected to
    the input width, is added to each neighbor's feature before the mean --
    the same additive edge-injection GINEConv uses, just under mean instead
    of sum aggregation.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__(aggr="mean")
        self.lin_edge = nn.Linear(edge_dim, in_dim)
        self.lin_l = nn.Linear(in_dim, out_dim)  # self
        self.lin_r = nn.Linear(in_dim, out_dim)  # aggregated neighbors

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        aggregated = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_l(x) + self.lin_r(aggregated)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return F.relu(x_j + self.lin_edge(edge_attr))


def _make_edge_conv(kind: str, in_dim: int, out_dim: int, heads: int,
                     edge_dim: int) -> nn.Module:
    if kind == "gine":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        return GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    if kind == "gat_edge":
        assert out_dim % heads == 0, "hidden must be divisible by heads"
        return GATConv(in_dim, out_dim // heads, heads=heads, edge_dim=edge_dim)
    if kind == "sage_edge":
        return SAGEEdgeConv(in_dim, out_dim, edge_dim)
    raise ValueError(f"unknown conv type: {kind!r} (expected one of {EDGE_MODELS})")


class EdgeGNNClassifier(nn.Module):
    def __init__(
        self,
        conv: str,
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        edge_dim: int = EDGE_DIM,
    ) -> None:
        super().__init__()
        if conv not in EDGE_MODELS:
            raise ValueError(f"unknown conv type: {conv!r} (expected one of {EDGE_MODELS})")
        self.conv_type = conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                _make_edge_conv(conv, in_dim if layer == 0 else hidden, hidden,
                                 heads, edge_dim)
            )
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = (
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        graph_repr = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.head(graph_repr).squeeze(-1)


def build_edge_model(conv: str, **kwargs) -> EdgeGNNClassifier:
    return EdgeGNNClassifier(conv=conv, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _shape_check() -> None:
    """Forward pass for all three edge-aware operators, including a
    zero-edge graph (the same regression test models.py/models_gine.py use)."""
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph

    graphs = [
        smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", label=1.0),
        smiles_to_graph("C", label=0.0),
        smiles_to_graph("CCO", label=1.0),
        smiles_to_graph("c1ccc2c(c1)ccc1ccccc12", label=0.0),
    ]
    batch = next(iter(DataLoader(graphs, batch_size=4)))
    assert batch.edge_index.shape[1] > 0, "expected some edges in the batch"
    print(f"batch: {batch.num_graphs} graphs, {batch.num_nodes} atoms, "
          f"{batch.edge_index.shape[1]} directed edges "
          f"(includes a zero-edge single-atom graph)\n")

    for kind in EDGE_MODELS:
        model = build_edge_model(kind)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4,), f"{kind}: expected (4,), got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{kind}: non-finite output"
        print(f"  {kind:<10} -> logits {tuple(out.shape)}  "
              f"params={count_parameters(model):,}")

    print("\nshape check passed (all three edge-aware operators handle the zero-edge graph)")


if __name__ == "__main__":
    _shape_check()
