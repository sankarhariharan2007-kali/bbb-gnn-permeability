"""Hierarchical Reasoning Model over the atom graph. Arm 1b.

HRM (Wang et al. 2025) couples two recurrent modules on different timescales:
a fast low-level module that iterates several times per cycle, and a slow
high-level module that updates once per cycle from the low-level module's
settled state. Both are weight-shared across all steps, so the network buys
EFFECTIVE DEPTH without parameters -- which is the only part of the paper
plausibly relevant at n=1572.

WHAT IS KEPT, and why these three:
  * two timescales      -- the mechanism being tested
  * weight sharing      -- the reason it could work at this data scale
  * one-step gradient   -- HRM backpropagates only the final cycle and runs
                           the rest under no_grad, so memory is O(1) in depth
                           rather than O(cycles). Without it this is just a
                           deep GNN with tied weights.

WHAT IS DROPPED: the published 27M parameter count (that is ~17,000 parameters
per training molecule here, against a GIN at ~121k already reaching ~0.89), ACT
halting (needs a halting target this task does not have), and deep supervision
across segments. This runs at **parameter parity with the base skeleton** so a
difference is attributable to the recurrence, not to capacity.

THE PRIOR IS AGAINST IT, and that is worth stating before the numbers arrive.
HRM was built for ARC/Sudoku/maze -- fixed-size symbolic puzzles with a latent
multi-step algorithm and ~1M augmented instances. BBB permeability has no
latent algorithm; LightGBM on 8 descriptors reaches 0.88. And this repo's 2x2
measured long-range atom connectivity at **-0.019** while readout width was
**+0.130**, so depth is the axis this task has already shown it does not care
about. This arm exists to measure that rather than assert it.

    python -m src.models_hrm        # shape + depth self-check
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_mean_pool

from .featurize import NODE_DIM
from .models import _make_conv, _make_norm

HRM_MODELS = ["gcn", "sage", "gin", "gat"]


class HRMGNN(nn.Module):
    def __init__(
        self,
        conv: str = "gin",
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        n_inner: int = 3,
        n_outer: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
        norm: str = "graph",
        virtual_node: bool = False,
        one_step_grad: bool = True,
    ) -> None:
        super().__init__()
        if conv not in HRM_MODELS:
            raise ValueError(f"unknown conv {conv!r}; expected one of {HRM_MODELS}")
        if virtual_node:
            raise NotImplementedError("virtual_node is implemented in models.py only")
        self.conv_type = conv
        self.n_inner, self.n_outer = n_inner, n_outer
        self.one_step_grad = one_step_grad

        self.embed = nn.Linear(in_dim, hidden)
        # One conv each, reused at every step. That is the weight sharing.
        self.conv_low = _make_conv(conv, 2 * hidden, hidden, heads)
        self.conv_high = _make_conv(conv, 2 * hidden, hidden, heads)
        self.norm_low, needs_batch = _make_norm(norm, hidden)
        self.norm_high, _ = _make_norm(norm, hidden)
        self.norm_needs_batch = needs_batch

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def _cycle(self, z_low, z_high, x_emb, edge_index, batch):
        """One outer cycle: n_inner fast updates, then one slow update."""
        for _ in range(self.n_inner):
            inp = torch.cat([z_low, z_high + x_emb], dim=1)   # input re-injected
            z_low = self.conv_low(inp, edge_index)
            z_low = (self.norm_low(z_low, batch) if self.norm_needs_batch
                     else self.norm_low(z_low))
            z_low = torch.relu(z_low)
        inp = torch.cat([z_high, z_low], dim=1)
        z_high = self.conv_high(inp, edge_index)
        z_high = (self.norm_high(z_high, batch) if self.norm_needs_batch
                  else self.norm_high(z_high))
        z_high = torch.relu(z_high)
        return z_low, z_high

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_emb = self.embed(x)
        z_low = torch.zeros_like(x_emb)
        z_high = torch.zeros_like(x_emb)

        # One-step gradient: every cycle but the last runs detached, so memory
        # does not grow with depth. The final cycle carries the gradient.
        if self.one_step_grad and self.n_outer > 1:
            with torch.no_grad():
                for _ in range(self.n_outer - 1):
                    z_low, z_high = self._cycle(z_low, z_high, x_emb,
                                                edge_index, batch)
            z_low, z_high = z_low.detach(), z_high.detach()
            z_low, z_high = self._cycle(z_low, z_high, x_emb, edge_index, batch)
        else:
            for _ in range(self.n_outer):
                z_low, z_high = self._cycle(z_low, z_high, x_emb,
                                            edge_index, batch)

        z_high = self.dropout(z_high)
        graph_repr = torch.cat([global_mean_pool(z_high, batch),
                                global_max_pool(z_high, batch)], dim=1)
        return self.head(graph_repr).squeeze(-1)


def build_hrm_model(conv: str, **kwargs) -> HRMGNN:
    return HRMGNN(conv=conv, **kwargs)


def _shape_check() -> None:
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph
    from .models import build_model, count_parameters

    graphs = [smiles_to_graph(s, label=1.0) for s in
              ["CC(=O)Oc1ccccc1C(=O)O", "C", "CCO", "c1ccc2c(c1)ccc1ccccc12"]]
    batch = next(iter(DataLoader(graphs, batch_size=4)))

    print("parameter parity against the flat base (the whole point):")
    for kind in HRM_MODELS:
        hrm, flat = build_hrm_model(kind), build_model(kind)
        ph, pf = count_parameters(hrm), count_parameters(flat)
        hrm.eval()
        with torch.no_grad():
            out = hrm(batch)
        assert out.shape == (4,), f"{kind}: got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{kind}: non-finite output"
        print(f"  {kind:<5} hrm={ph:>8,}  flat={pf:>8,}  ratio={ph/pf:.2f}")

    # Effective depth: n_inner * n_outer message-passing steps from the
    # parameters of two convs. If this were not true the arm is just a deep GNN.
    m = build_hrm_model("gin", n_inner=3, n_outer=2)
    print(f"\neffective depth: {m.n_inner * m.n_outer} message-passing steps "
          f"from 2 convs' worth of weights (flat model: 3 steps from 3 convs)")

    # The one-step gradient must actually detach: gradient flows, but memory
    # does not grow with n_outer. Check the gradient reaches the shared convs.
    m.train()
    loss = m(batch).sum()
    loss.backward()
    assert m.conv_high.nn[0].weight.grad is not None, "no gradient to conv_high"
    assert m.conv_low.nn[0].weight.grad is not None, "no gradient to conv_low"
    assert m.embed.weight.grad is not None, "no gradient to the input embedding"
    print("one-step gradient: reaches embed, conv_low and conv_high")

    # Deeper recurrence must not change the parameter count. That is the claim.
    a = count_parameters(build_hrm_model("gin", n_inner=3, n_outer=2))
    b = count_parameters(build_hrm_model("gin", n_inner=8, n_outer=4))
    assert a == b, f"depth changed the parameter count: {a} vs {b}"
    print(f"depth is free: n_inner=3,n_outer=2 and n_inner=8,n_outer=4 "
          f"both have {a:,} parameters")

    # Batch independence, same assertion the other families carry.
    alone = next(iter(DataLoader([graphs[0]], batch_size=1)))
    m2 = build_hrm_model("gin"); m2.eval()
    with torch.no_grad():
        drift = float((m2(alone)[0] - m2(batch)[0]).abs())
    assert drift < 1e-5, f"logit moved by {drift:.2e} with batch composition"
    print(f"batch independence: drift {drift:.2e}")

    print("\nmodels_hrm shape check passed")


if __name__ == "__main__":
    _shape_check()
