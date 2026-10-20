"""One GNN skeleton, four convolution operators.

Depth, hidden width, normalization, readout, classifier head and training loop
are identical across all four models. Only the aggregation scheme differs, so a
performance gap is attributable to the operator rather than to incidental
capacity differences.

NORMALIZATION IS A SWITCH, AND THE DEFAULT CHANGED. BatchNorm1d makes a
molecule's logit depend on which other molecules share its batch. For a
classifier that is a mild training detail; for this project it is a correctness
problem, because the same network is later frozen and used as a reward. Under
BatchNorm in train mode the same molecule earns a different reward depending on
which candidates happened to land in its GRPO group -- the reward stops being a
function of the molecule at all. `reward.py` and `train_gan.py` each carry an
`eval()` guard and a self-check for exactly this. LayerNorm (the default) is
batch-independent by construction, so the hazard is gone rather than guarded.

`eval()` is still load-bearing: `nn.Dropout` keeps the forward pass stochastic
in train mode. The guards stay, for a smaller reason than before.

VIRTUAL NODE. With `num_layers=3` an atom's embedding only ever sees atoms
within 3 bonds, so on a 40-heavy-atom drug the two ends of the molecule are
mutually invisible -- yet BBB permeability is driven by whole-molecule TPSA,
logP and size. The mean+max readout gives a global view at the *output*, but
the convolutions cannot condition on it. A virtual node carries a per-graph
state that is injected into every atom between layers, making the effective
diameter 2 at any molecule size. Preferred over simply adding layers, which
oversmooths. Off by default so it stays an ablatable arm.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, GraphNorm, SAGEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .featurize import NODE_DIM

MODELS = ["gcn", "sage", "gin", "gat"]
NORMS = ["layer", "graph", "batch"]


def _make_norm(kind: str, hidden: int) -> tuple[nn.Module, bool]:
    """Return (module, needs_batch). Shared by every model family."""
    if kind == "layer":
        return nn.LayerNorm(hidden), False
    if kind == "graph":
        return GraphNorm(hidden), True
    if kind == "batch":
        # Kept so the pre-LayerNorm results can be reproduced and compared.
        return nn.BatchNorm1d(hidden), False
    raise ValueError(f"unknown norm: {kind!r} (expected one of {NORMS})")


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
        norm: str = "graph",
        virtual_node: bool = False,
    ) -> None:
        super().__init__()
        self.conv_type = conv
        self.norm_type = norm
        self.virtual_node = virtual_node
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                _make_conv(conv, in_dim if layer == 0 else hidden, hidden, heads)
            )
            module, needs_batch = _make_norm(norm, hidden)
            self.norms.append(module)
        self.norm_needs_batch = needs_batch
        self.dropout = nn.Dropout(dropout)

        # Mean + max readout: mean captures average atom environment, max picks
        # up whether any single strong substructure is present.
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Built LAST on purpose. These parameters draw from the RNG, so
        # constructing them earlier would shift every subsequent weight and the
        # virtual-node arm would differ from the baseline by its seed as well
        # as by its architecture -- which is what the no-op self-check caught.
        if virtual_node:
            # Injected *between* layers, where x is already `hidden`-wide, so
            # no input projection is needed.
            self.vn_embedding = nn.Parameter(torch.zeros(1, hidden))
            self.vn_mlp = nn.ModuleList(
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                              nn.Linear(hidden, hidden))
                for _ in range(num_layers - 1)
            )
            # Zeroing the OUTPUT layer, not just the embedding: a zero input
            # through a randomly initialised MLP still emits its biases, so
            # without this the arm starts as a perturbation of the baseline
            # rather than as the baseline.
            for mlp in self.vn_mlp:
                nn.init.zeros_(mlp[-1].weight)
                nn.init.zeros_(mlp[-1].bias)

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        vn = None
        if self.virtual_node:
            n_graphs = int(batch.max()) + 1 if batch.numel() else 1
            vn = self.vn_embedding.expand(n_graphs, -1)

        last = len(self.convs) - 1
        for layer, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x, batch) if self.norm_needs_batch else norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            if vn is not None and layer < last:
                # Read every atom into the graph-level state, then write it
                # back to every atom: one hop to anywhere in the molecule.
                vn = vn + self.vn_mlp[layer](global_add_pool(x, batch))
                x = x + vn[batch]

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

    print("\nnorms:")
    for norm in NORMS:
        model = build_model("gin", norm=norm)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out).all(), f"norm={norm}: non-finite output"
        print(f"  {norm:<5} -> logits {tuple(out.shape)}  "
              f"params={count_parameters(model):,}")

    print("\nvirtual node:")
    for kind in MODELS:
        model = build_model(kind, virtual_node=True)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4,), f"{kind}+vn: got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{kind}+vn: non-finite output"
    # Zero-initialised, so an untrained virtual node must be a no-op: the arm
    # starts from the baseline rather than from a perturbation of it.
    torch.manual_seed(0)
    plain = build_model("gin")
    torch.manual_seed(0)
    withvn = build_model("gin", virtual_node=True)
    plain.eval(), withvn.eval()
    with torch.no_grad():
        assert torch.allclose(plain(batch), withvn(batch), atol=1e-6), (
            "an untrained virtual node must not change the output"
        )
    print(f"  all four forward; zero-init vn is a no-op "
          f"(+{count_parameters(withvn) - count_parameters(plain):,} params)")

    # The reason the default moved off BatchNorm. A molecule's logit must not
    # depend on what shares its batch, or the frozen reward is not a function
    # of the molecule. This is the cheap version of reward.py's own assertion.
    print("\nbatch independence (eval mode):")
    alone = next(iter(DataLoader([graphs[0]], batch_size=1)))
    for norm in NORMS:
        model = build_model("gin", norm=norm)
        model.eval()
        with torch.no_grad():
            solo = model(alone)[0]
            crowded = model(batch)[0]
        drift = float((solo - crowded).abs())
        assert drift < 1e-5, f"norm={norm}: logit moved by {drift:.2e} with batch"
        print(f"  {norm:<5} -> drift {drift:.2e}")

    # Train mode is where BatchNorm diverges; LayerNorm must not. Dropout is
    # disabled here so the comparison isolates the normalization.
    print("\nbatch independence (train mode, dropout off):")
    for norm in NORMS:
        model = build_model("gin", norm=norm, dropout=0.0)
        model.train()
        with torch.no_grad():
            drift = float((model(alone)[0] - model(batch)[0]).abs())
        verdict = "STATIONARY" if drift < 1e-5 else "batch-dependent"
        print(f"  {norm:<5} -> drift {drift:.2e}  {verdict}")
        if norm != "batch":
            assert drift < 1e-5, f"norm={norm} must be batch-independent, got {drift:.2e}"

    print("\nshape check passed (all four handle the zero-edge graph)")


if __name__ == "__main__":
    _shape_check()
