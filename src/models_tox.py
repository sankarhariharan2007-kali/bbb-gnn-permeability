"""Multi-task GNN for Tox21: 12 assay heads over the shared graph encoder.

Additive file, same convention as models_edge.py and models_hybrid.py: the
base four-operator comparison in models.py is untouched. The encoder is
duplicated rather than imported-and-patched so that a change to the toxicity
model can never silently alter the permeability classifier the reward already
depends on.

The only structural difference from models.GNNClassifier is the head width:
12 logits instead of 1, one per Tox21 assay. Multi-task is the right shape
here because the assays are sparsely labelled -- no compound is measured in
all 12 -- so a shared encoder lets each assay borrow statistical strength
from the others rather than training 12 small models on 12 small subsets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool

from .featurize import NODE_DIM
from .models import MODELS, _make_conv, _make_norm

# The 12 Tox21 assays, in the column order of the released CSV. NR- are
# nuclear-receptor signalling panels, SR- are stress-response panels.
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


class MultiTaskGNN(nn.Module):
    def __init__(
        self,
        conv: str = "gin",
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        heads: int = 4,
        n_tasks: int = len(TOX21_TASKS),
        norm: str = "graph",
        virtual_node: bool = False,
    ) -> None:
        super().__init__()
        if conv not in MODELS:
            raise ValueError(f"unknown conv {conv!r}; expected one of {sorted(MODELS)}")
        self.conv_type = conv
        self.n_tasks = n_tasks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                _make_conv(conv, in_dim if layer == 0 else hidden, hidden, heads)
            )
            module, needs_batch = _make_norm(norm, hidden)
            self.norms.append(module)
        self.norm_needs_batch = needs_batch
        if virtual_node:
            # Scoped to the base family (models.py): the virtual-node arm is an
            # ablation against the four plain operators, and adding it here too
            # would make that comparison span two families at once.
            raise NotImplementedError(
                "virtual_node is implemented in models.py only; this family "
                "must be reloaded with virtual_node=False"
            )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x, batch) if self.norm_needs_batch else norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        graph_repr = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.head(graph_repr)  # (batch, n_tasks) logits


def build_tox_model(conv: str = "gin", **kwargs) -> MultiTaskGNN:
    return MultiTaskGNN(conv=conv, **kwargs)


def masked_bce(logits: torch.Tensor, targets: torch.Tensor,
               mask: torch.Tensor, pos_weight: torch.Tensor | None = None
               ) -> torch.Tensor:
    """BCE over observed labels only.

    Tox21 is sparsely labelled -- every compound is missing several assays --
    so an unmasked loss would train the model to predict whatever value the
    missing entries were filled with. The mask is not an optimization; it is
    the difference between learning toxicity and learning the fill value.

    pos_weight compensates the class imbalance, which runs from 2.9% positive
    (NR-PPAR-gamma) to 16.2% (SR-ARE). Without it the model can reach high
    accuracy by predicting "non-toxic" everywhere, which is exactly the
    failure that would make this term useless as a reward penalty.
    """
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=mask, reduction="sum", pos_weight=pos_weight
    )
    return loss / mask.sum().clamp(min=1.0)


def _shape_check() -> None:
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph

    graphs = [smiles_to_graph(s) for s in
              ("CCO", "c1ccccc1", "ClC(Cl)(Cl)Cl", "CC(=O)Oc1ccccc1C(=O)O")]
    batch = next(iter(DataLoader([g for g in graphs if g], batch_size=4)))

    for conv in sorted(MODELS):
        out = build_tox_model(conv)(batch)
        assert out.shape == (4, 12), f"{conv} gave {out.shape}, expected (4, 12)"

    # Masked loss must ignore unobserved entries entirely: changing a target
    # under a zero mask must not change the loss.
    torch.manual_seed(0)
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 2, (4, 12)).float()
    mask = torch.ones(4, 12)
    mask[0, :6] = 0.0
    a = masked_bce(logits, targets, mask)
    poisoned = targets.clone()
    poisoned[0, :6] = 1.0 - poisoned[0, :6]
    b = masked_bce(logits, poisoned, mask)
    assert torch.allclose(a, b), "masked entries leaked into the loss"

    # And an all-zero mask must not divide by zero.
    assert torch.isfinite(masked_bce(logits, targets, torch.zeros(4, 12)))

    print(f"models_tox shape check passed ({len(TOX21_TASKS)} tasks)")


if __name__ == "__main__":
    _shape_check()
