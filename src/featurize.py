"""SMILES -> PyTorch Geometric graph conversion via RDKit.

The only input signal in this project is the SMILES string, so every feature
here is derived from the RDKit molecular graph. No descriptors, no fingerprints.
"""

from __future__ import annotations

import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

# RDKit is chatty about the unparseable SMILES in BBBP; we count them ourselves.
RDLogger.DisableLog("rdApp.*")


ATOM_SYMBOLS = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Every one-hot carries a trailing "other" bucket, so an unseen value can never
# silently collide with a real category.
NODE_DIM = (
    len(ATOM_SYMBOLS) + 1
    + len(DEGREES) + 1
    + len(FORMAL_CHARGES) + 1
    + len(NUM_HS) + 1
    + len(HYBRIDIZATIONS) + 1
    + 2  # is_aromatic, is_in_ring
)
EDGE_DIM = len(BOND_TYPES) + 1 + 2  # bond type + other, is_conjugated, is_in_ring


def _one_hot(value, choices: list) -> list[float]:
    """One-hot over `choices` with a trailing catch-all slot."""
    encoding = [0.0] * (len(choices) + 1)
    try:
        encoding[choices.index(value)] = 1.0
    except ValueError:
        encoding[-1] = 1.0
    return encoding


def atom_features(atom: Chem.rdchem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.rdchem.Bond) -> list[float]:
    """Stored on Data.edge_attr but deliberately unused by the four models.

    GCN and GraphSAGE cannot consume edge features, so feeding them to GIN/GAT
    alone would confound the architecture comparison with an input advantage.
    Keeping them here makes an edge-aware ablation (GINEConv) a small change.
    """
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def smiles_to_graph(smiles: str, label: float | None = None) -> Data | None:
    """Return a PyG `Data`, or None if RDKit cannot parse the SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    # Undirected graphs are stored as both directions per PyG convention.
    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feats, feats]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.float)
    else:
        # Single-atom molecules ("C", "O") appear in both datasets. The empty
        # tensors must still carry the right shape/dtype or batching crashes.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = Chem.MolToSmiles(mol)  # canonical form
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    return data


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)


def _smoke_test() -> None:
    print(f"NODE_DIM={NODE_DIM}  EDGE_DIM={EDGE_DIM}")

    aspirin = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O", label=1.0)
    assert aspirin is not None
    assert aspirin.x.shape == (13, NODE_DIM), aspirin.x.shape
    assert aspirin.edge_index.shape[1] == 26, aspirin.edge_index.shape
    print(f"aspirin      -> {aspirin.num_nodes} atoms, "
          f"{aspirin.edge_index.shape[1]} directed edges, y={aspirin.y.item()}")

    methane = smiles_to_graph("C", label=0.0)
    assert methane is not None
    assert methane.edge_index.shape == (2, 0), methane.edge_index.shape
    assert methane.edge_index.dtype == torch.long
    assert methane.edge_attr.shape == (0, EDGE_DIM)
    print(f"methane 'C'  -> {methane.num_nodes} atom, "
          f"edge_index.shape={tuple(methane.edge_index.shape)} (zero-edge graph)")

    bad = smiles_to_graph("this-is-not-a-molecule")
    assert bad is None
    print("invalid SMILES -> None")

    print("\nsmoke test passed")


if __name__ == "__main__":
    _smoke_test()
