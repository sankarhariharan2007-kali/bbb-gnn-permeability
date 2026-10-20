"""Hierarchical two-level GNN: atoms inside fragments, then fragments.

Arm 1a. The flat models message-pass over one atom graph; with num_layers=3 an
atom sees three bonds out, and the molecule's organisation into functional
groups is nowhere in the architecture. This model puts it there:

    level 1   atom GNN, restricted to INTRA-fragment bonds -> pool per fragment
    level 2   motif GNN over the BRICS junction graph      -> pool per molecule

The claim being tested is that permeability is governed by functional groups
rather than by isolated atoms -- a carboxylic acid forbids passive diffusion as
a unit, and in a flat graph that fact is smeared over three atoms and diluted
by message passing. Here it is one node.

NO FRAGMENT VOCABULARY, DELIBERATELY. A pure motif-level model would embed each
fragment by its identity, which needs a vocabulary -- and a vocabulary built
over the whole dataset is transductive leakage under a scaffold split, since
the split exists to give test molecules unseen cores and a global vocabulary
hands every test fragment its own embedding slot. Learning the fragment vector
from its atoms sidesteps that entirely: an unseen fragment is embedded by the
same atom encoder as a seen one, so there is no OOV case and nothing to leak.

THE FEATURIZER IS NOT TOUCHED. `frag_id` and the motif edges are derived here
from `data.smiles` via `brics.atom_fragments`, so `FEATURE_ID` does not move
and no committed checkpoint is invalidated by this family existing.

    python -m src.models_brics        # shape + batching self-check
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from .brics import atom_fragments
from .featurize import NODE_DIM
from .models import _make_conv, _make_norm

RDLogger.DisableLog("rdApp.*")

BRICS_MODELS = ["gcn", "sage", "gin", "gat"]


class MotifData(Data):
    """Carries a second graph, so PyG must be told how to batch it.

    `motif_edge_index` and `frag_id` index FRAGMENTS, not atoms, so on batching
    they must be offset by the running fragment count rather than the atom
    count. Without this override both silently point into the wrong graph's
    fragments once batch_size > 1 -- and it does not raise, it just trains on
    nonsense.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("motif_edge_index", "frag_id"):
            return int(self.num_frags)
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "motif_edge_index":
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def to_motif_data(data: Data) -> MotifData | None:
    """Decorate an atom graph with its fragment assignment and motif edges."""
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None
    frag_id, edges = atom_fragments(mol)
    if len(frag_id) != data.num_nodes:
        return None  # canonicalization disagreement; skip rather than mismatch

    out = MotifData(x=data.x, edge_index=data.edge_index,
                    edge_attr=data.edge_attr)
    out.smiles = data.smiles
    if hasattr(data, "y"):
        out.y = data.y
    out.frag_id = torch.tensor(frag_id, dtype=torch.long)
    out.num_frags = len(set(frag_id))
    if edges:
        src = [a for a, b in edges] + [b for a, b in edges]   # undirected
        dst = [b for a, b in edges] + [a for a, b in edges]
        out.motif_edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        out.motif_edge_index = torch.empty((2, 0), dtype=torch.long)
    return out


class MotifGNN(nn.Module):
    def __init__(
        self,
        conv: str = "gin",
        in_dim: int = NODE_DIM,
        hidden: int = 128,
        atom_layers: int = 2,
        motif_layers: int = 2,
        dropout: float = 0.3,
        heads: int = 4,
        norm: str = "graph",
        virtual_node: bool = False,
        intra_only: bool = False,
        fuse: bool = True,
    ) -> None:
        """`intra_only` / `fuse` select the two arms this class can be.

        `intra_only=True, fuse=False` is arm 1a as first built: level 1 sees
        ONLY intra-fragment bonds, so the ~4 BRICS cut bonds per molecule are
        deleted from the atom graph and survive only as coarse fragment-level
        edges. Measured on BBBP over 10 paired seeds, that costs
        **-0.1177 +/- 0.0103** against the flat base -- about 3x the seed
        noise, in all four operators. The junctions BRICS cuts are amides,
        esters and aryl-aryl couplings, i.e. the pharmacophores, so removing
        them from atom-level message passing removes the signal.

        `intra_only=False, fuse=True` (the default) is the repair: level 1 runs
        over the FULL atom graph, exactly as the flat model does, and the motif
        graph is an ADDITIONAL readout channel concatenated at the head. That
        makes the family strictly additive -- the atom pathway is the base
        model, and the head can learn to ignore the motif half -- so it tests
        whether fragment structure ADDS anything rather than whether it can
        replace atom-level detail.
        """
        super().__init__()
        if conv not in BRICS_MODELS:
            raise ValueError(f"unknown conv {conv!r}; expected one of {BRICS_MODELS}")
        if virtual_node:
            raise NotImplementedError(
                "virtual_node is implemented in models.py only; the motif graph "
                "already gives every fragment a short path to every other"
            )
        self.conv_type = conv
        self.norm_type = norm
        self.intra_only = intra_only
        self.fuse = fuse

        # Level 1 -- atoms, intra-fragment edges only.
        self.atom_convs, self.atom_norms = nn.ModuleList(), nn.ModuleList()
        for layer in range(atom_layers):
            self.atom_convs.append(
                _make_conv(conv, in_dim if layer == 0 else hidden, hidden, heads))
            module, needs_batch = _make_norm(norm, hidden)
            self.atom_norms.append(module)
        self.norm_needs_batch = needs_batch

        # A fragment is summarised by mean+max over its atoms, mirroring the
        # molecule-level readout the flat models use.
        self.frag_proj = nn.Linear(2 * hidden, hidden)

        # Level 2 -- fragments, BRICS junction edges.
        self.motif_convs, self.motif_norms = nn.ModuleList(), nn.ModuleList()
        for _ in range(motif_layers):
            self.motif_convs.append(_make_conv(conv, hidden, hidden, heads))
            module, _ = _make_norm(norm, hidden)
            self.motif_norms.append(module)

        self.dropout = nn.Dropout(dropout)
        # fuse -> [atom mean, atom max, motif mean, motif max]
        head_in = (4 if fuse else 2) * hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )

    def forward(self, data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        frag_id, motif_edge_index = data.frag_id, data.motif_edge_index

        # arm 1a kept only intra-fragment bonds; the default keeps them all,
        # so the atom pathway matches the flat model exactly.
        if self.intra_only and edge_index.numel():
            intra = frag_id[edge_index[0]] == frag_id[edge_index[1]]
            atom_edge_index = edge_index[:, intra]
        else:
            atom_edge_index = edge_index

        for conv, norm in zip(self.atom_convs, self.atom_norms):
            x = conv(x, atom_edge_index)
            x = norm(x, batch) if self.norm_needs_batch else norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        n_frags = int(frag_id.max()) + 1 if frag_id.numel() else 0
        h = torch.cat([global_mean_pool(x, frag_id, size=n_frags),
                       global_max_pool(x, frag_id, size=n_frags)], dim=1)
        h = torch.relu(self.frag_proj(h))

        # Which molecule each fragment belongs to, derived from its atoms.
        frag_batch = torch.zeros(n_frags, dtype=torch.long, device=x.device)
        frag_batch.scatter_(0, frag_id, batch)

        for conv, norm in zip(self.motif_convs, self.motif_norms):
            h = conv(h, motif_edge_index)
            h = norm(h, frag_batch) if self.norm_needs_batch else norm(h)
            h = torch.relu(h)
            h = self.dropout(h)

        motif_repr = torch.cat([global_mean_pool(h, frag_batch),
                                global_max_pool(h, frag_batch)], dim=1)
        if self.fuse:
            atom_repr = torch.cat([global_mean_pool(x, batch),
                                   global_max_pool(x, batch)], dim=1)
            motif_repr = torch.cat([atom_repr, motif_repr], dim=1)
        return self.head(motif_repr).squeeze(-1)


def build_brics_model(conv: str, **kwargs) -> MotifGNN:
    return MotifGNN(conv=conv, **kwargs)


def _shape_check() -> None:
    from torch_geometric.loader import DataLoader

    from .featurize import smiles_to_graph
    from .models import count_parameters

    smis = [
        "CC(=O)Oc1ccccc1C(=O)O",            # 4 fragments
        "c1ccc2c(c1)ccc1ccccc12",           # pyrene: zero cuts, 1 fragment
        "C",                                 # single atom, zero edges
        "CCN1CCC[C@H]1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC",
    ]
    graphs = [to_motif_data(smiles_to_graph(s, label=1.0)) for s in smis]
    assert all(g is not None for g in graphs)
    for s, g in zip(smis, graphs):
        print(f"  {g.num_nodes:>3} atoms -> {g.num_frags:>2} fragments, "
              f"{g.motif_edge_index.shape[1] // 2:>2} junctions   {s[:40]}")

    batch = next(iter(DataLoader(graphs, batch_size=4)))

    # Batching is the thing that breaks silently, so check it explicitly:
    # fragment ids must be offset per graph, not restarted at 0.
    assert int(batch.frag_id.max()) + 1 == sum(g.num_frags for g in graphs), (
        f"frag_id not offset on batching: max={int(batch.frag_id.max())}, "
        f"expected {sum(g.num_frags for g in graphs) - 1}"
    )
    assert int(batch.motif_edge_index.max()) < sum(g.num_frags for g in graphs)
    print(f"\nbatched: {batch.num_graphs} graphs, {batch.num_nodes} atoms, "
          f"{int(batch.frag_id.max()) + 1} fragments "
          f"(ids correctly offset across the batch)\n")

    for kind in BRICS_MODELS:
        model = build_brics_model(kind)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (4,), f"{kind}: got {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{kind}: non-finite output"
        print(f"  {kind:<5} -> logits {tuple(out.shape)}  "
              f"params={count_parameters(model):,}")

    # A molecule scored alone must match the same molecule scored in company,
    # or the reward built on this family would not be a function of the
    # molecule. Same assertion the flat models carry.
    alone = next(iter(DataLoader([graphs[0]], batch_size=1)))
    model = build_brics_model("gin")
    model.eval()
    with torch.no_grad():
        drift = float((model(alone)[0] - model(batch)[0]).abs())
    assert drift < 1e-5, f"logit moved by {drift:.2e} with batch composition"
    print(f"\nbatch independence: drift {drift:.2e}")

    print("\nmodels_brics shape check passed")


if __name__ == "__main__":
    _shape_check()
