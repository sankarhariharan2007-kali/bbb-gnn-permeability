"""
Priority 3: D-MPNN (Chemprop's directed bond-level message passing, Yang et
al. 2019) as one more operator in the comparison.

This is architecturally different from the four models in models.py -- it
passes messages along DIRECTED BONDS rather than atoms, which is the reason
it's the closest published baseline to this exact setup and consistently
beats plain GCN/GAT/GIN/SAGE on BBBP-style tasks in the literature. Depth,
hidden width, dropout, and the mean+max readout + 2-layer head are kept
identical to the rest of the project so the comparison stays about the
message-passing mechanism, not incidental capacity differences; only the
internal update rule is D-MPNN's own, since that IS the thing being tested.

Directed-bond bookkeeping
-------------------------
src/featurize.py stores every bond as two directed edges, appended as a
consecutive (i, j), (j, i) pair (see `bond_features` callsite in
smiles_to_graph). PyG's default batching concatenates each graph's edge_index
as a contiguous block and never reorders within it, and every graph
contributes an even number of edges (bonds always emit two directed edges),
so the offset added to a later graph's edge indices is always even. Both
facts together mean the "edges 2k and 2k+1 are mutual reverses" invariant
survives batching intact, and `_reverse_edge_index` below only has to encode
that fixed pairing -- it does not need per-graph bookkeeping.

D-MPNN update, standard formulation:
    h_uv^0        = ReLU(W_i [x_u || e_uv])
    msg_uv^t      = sum over k in N(u), k != v, of h_ku^{t-1}
                  = (sum of h over all edges landing on u) - h_vu^{t-1}
    h_uv^t        = ReLU(h_uv^0 + W_h msg_uv^t)
    m_v           = sum_{k in N(v)} h_kv^{T-1}
    atom_v        = ReLU(W_o [x_v || m_v])
Then mean+max pool over atoms -> the same head shape as the rest of the repo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool
from torch_geometric.utils import scatter

from .featurize import EDGE_DIM, NODE_DIM

DMPNN_MODELS = ["dmpnn"]


def _reverse_edge_index(num_edges: int, device) -> torch.Tensor:
    """rev[i] = index of the mutual-reverse edge of edge i.

    Relies on src/featurize.py's invariant that bonds are appended as
    consecutive (i,j),(j,i) pairs, which -- as argued in the module
    docstring -- survives PyG batching unchanged.
    """
    rev = torch.empty(num_edges, dtype=torch.long, device=device)
    rev[0::2] = torch.arange(1, num_edges, 2, device=device)
    rev[1::2] = torch.arange(0, num_edges, 2, device=device)
    return rev


class DMPNNEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden: int, depth: int,
                 dropout: float) -> None:
        super().__init__()
        self.depth = depth
        self.w_i = nn.Linear(in_dim + edge_dim, hidden)
        self.w_h = nn.Linear(hidden, hidden)
        self.w_o = nn.Linear(in_dim + hidden, hidden)
        self.bn_edge = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(depth - 1))
        self.bn_atom = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        src, dst = edge_index[0], edge_index[1]

        if num_edges == 0:
            # No bonds at all (e.g. a lone "C"): every atom's incoming
            # message is zero, skip straight to the atom update.
            atom_msg = x.new_zeros((num_nodes, self.w_o.out_features))
            atom_hidden = F.relu(self.w_o(torch.cat([x, atom_msg], dim=1)))
            return self.dropout(self.bn_atom(atom_hidden))

        rev = _reverse_edge_index(num_edges, x.device)

        h0 = F.relu(self.w_i(torch.cat([x[src], edge_attr], dim=1)))
        h = h0
        for t in range(self.depth - 1):
            incoming_sum = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
            msg = incoming_sum[src] - h[rev]
            h = F.relu(h0 + self.w_h(msg))
            h = self.dropout(self.bn_edge[t](h))

        atom_msg = scatter(h, dst, dim=0, dim_size=num_nodes, reduce="sum")
        atom_hidden = F.relu(self.w_o(torch.cat([x, atom_msg], dim=1)))
        return self.dropout(self.bn_atom(atom_hidden))


class DMPNNClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden: int = 128,
        depth: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = DMPNNEncoder(in_dim, edge_dim, hidden, depth, dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        atom_hidden = self.encoder(data.x, data.edge_index, data.edge_attr)
        graph_repr = torch.cat(
            [global_mean_pool(atom_hidden, data.batch),
             global_max_pool(atom_hidden, data.batch)], dim=1
        )
        return self.head(graph_repr).squeeze(-1)


def build_dmpnn_model(**kwargs) -> DMPNNClassifier:
    return DMPNNClassifier(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _shape_check() -> None:
    """Forward pass including a zero-edge graph, mirroring models.py's check."""
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph

    graphs = [
        smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", label=1.0),
        smiles_to_graph("C", label=0.0),
        smiles_to_graph("CCO", label=1.0),
        smiles_to_graph("c1ccc2c(c1)ccc1ccccc12", label=0.0),
    ]
    batch = next(iter(DataLoader(graphs, batch_size=4)))
    model = build_dmpnn_model()
    model.eval()
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (4,), f"expected (4,), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite output"
    print(f"dmpnn -> logits {tuple(out.shape)}  params={count_parameters(model):,}")

    # Also check a batch that is ENTIRELY zero-edge graphs, since that's the
    # one case the reverse-edge-index trick could break silently on.
    lone_atoms = [smiles_to_graph("C", label=0.0), smiles_to_graph("O", label=1.0)]
    batch2 = next(iter(DataLoader(lone_atoms, batch_size=2)))
    with torch.no_grad():
        out2 = model(batch2)
    assert out2.shape == (2,) and torch.isfinite(out2).all()
    print("all-zero-edge batch -> OK")
    print("\nshape check passed")


if __name__ == "__main__":
    _shape_check()
