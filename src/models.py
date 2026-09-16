"""One GNN skeleton, four convolution operators.

Depth, hidden width, normalization, readout, classifier head and training loop
are identical across all four models. Only the aggregation scheme differs, so a
performance gap is attributable to the operator rather than to incidental
capacity differences.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn import global_max_pool, global_mean_pool

from .featurize import NODE_DIM

MODELS = ["gcn", "sage", "gin", "gat"]


def _make_conv(kind: str, in_dim: int, out_dim: int, heads: int) -> nn.Module:
    if kind == "gcn":
        return GCNConv(in_dim, out_dim)
    if kind == "sage":
        return SAGEConv(in_dim, out_dim)
    if kind == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        return GINConv(mlp, train_eps=True)
    if kind == "gat":
        # out_dim // heads keeps the concatenated output at out_dim, so GAT's
        # parameter count stays comparable instead of inflating heads-fold.
        assert out_dim % heads == 0, "hidden must be divisible by heads"
        return GATConv(in_dim, out_dim // heads, heads=heads)
    raise ValueError(f"unknown conv type: {kind!r} (expected one of {MODELS})")


class GNNClassifier(nn.Module):
    def __init__(
        self,
        conv: str,
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.conv_type = conv
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                _make_conv(conv, in_dim if layer == 0 else hidden, hidden, heads)
            )
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = nn.Dropout(dropout)

        # Mean + max readout: mean captures average atom environment, max picks
        # up whether any single strong substructure is present.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
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
        return self.head(graph_repr).squeeze(-1)  # single logit per molecule


def build_model(conv: str, **kwargs) -> GNNClassifier:
    return GNNClassifier(conv=conv, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _shape_check() -> None:
    """Forward pass for all four operators on a batch containing a zero-edge graph."""
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph

    # "C" is a single atom with no bonds; both datasets contain such molecules.
    graphs = [
        smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", label=1.0),
        smiles_to_graph("C", label=0.0),
        smiles_to_graph("CCO", label=1.0),
        smiles_to_graph("c1ccc2c(c1)ccc1ccccc12", label=0.0),
    ]
    batch = next(iter(DataLoader(graphs, batch_size=4)))
    assert (batch.edge_index.shape[1] > 0), "expected some edges in the batch"
    print(f"batch: {batch.num_graphs} graphs, {batch.num_nodes} atoms, "
          f"{batch.edge_index.shape[1]} directed edges "
          f"(includes a zero-edge single-atom graph)\n")

    for kind in MODELS:
        model = build_model(kind)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4,), f"{kind}: expected (4,), got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{kind}: non-finite output"
        print(f"  {kind:<5} -> logits {tuple(out.shape)}  "
              f"params={count_parameters(model):,}")

    print("\nshape check passed (all four handle the zero-edge graph)")


if __name__ == "__main__":
    _shape_check()
