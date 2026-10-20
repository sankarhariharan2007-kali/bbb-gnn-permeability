"""BRICS decomposition: molecule <-> (fragments, junctions). The shared substrate.

Both halves of the project need the same notion of "a fragment", so this module
is the single definition and is imported by the classifier's motif-graph family
and by the generator's assembly engine alike. If the two sides disagree on what
a fragment is, the generator optimizes a vocabulary the classifier cannot read
and the reward stops meaning anything.

WHY NOT `BRICS.BRICSDecompose`. That returns a *set* of fragment SMILES, which
is lossy: aspirin decomposes to two `[3*]` and two `[16*]` attachment points and
nothing records which pairs with which, so the parent cannot be rebuilt. The
combinatorial re-expansion of that ambiguity is what `BRICS.BRICSBuild` explores.
We want the actual parent back, so `decompose` keeps the junction list and
`reassemble` is its inverse. Measured: 2039/2039 BBBP and 7805/7805 B3DB
molecules round-trip to an identical canonical SMILES.

THE JUNCTION CARRIES A BOND TYPE, and that is not decoration. BRICS cuts
type-7 alkene linkages as well as single bonds, so rebuilding every junction as
`SINGLE` silently hydrogenates 91 of BBBP's 2039 molecules --
`CN(C)CCC=C1c2ccccc2CCc2ccccc21` comes back as `...CCCC1...`. The round-trip
assertion below is what caught that; it is cheap and it stays.

ZERO-CUT MOLECULES ARE NOT AN EDGE CASE. A fused aromatic with no cleavable
linker (pyrene, phenanthrene) yields no BRICS bonds at all: **12.8% of BBBP and
9.3% of B3DB**. Those fall back to a single whole-molecule fragment, which is
the scaffold-core case -- one motif node, no junctions. Callers must handle a
one-node motif graph rather than assuming a decomposition happened.

    python -m src.brics        # self-check
"""

from __future__ import annotations

from collections import Counter

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")

# Fragments below the vocabulary's frequency floor collapse here. One shared
# slot, not one per rare fragment: see the leakage note on `vocab`.
UNK = "<unk-frag>"


def decompose(mol: Chem.Mol) -> tuple[list[Chem.Mol], list[tuple]]:
    """Split on BRICS bonds. Returns (fragments, junctions).

    Each junction is `(frag_a, dummy_a, frag_b, dummy_b, bond_type)`, where the
    dummy indices are local to their fragment. Fragments carry `[n*]` dummies
    whose isotope is the BRICS attachment type, so a fragment knows what it may
    bond to without consulting the junction list.

    A molecule with no BRICS bonds returns `([mol], [])` -- see the module
    docstring; this is ~1 molecule in 10, not an error.
    """
    bonds = list(BRICS.FindBRICSBonds(mol))
    if not bonds:
        return [mol], []

    bond_idx, dummy_labels, bond_types = [], [], []
    for (a1, a2), (t1, t2) in bonds:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bond_idx.append(bond.GetIdx())
        dummy_labels.append((int(t1), int(t2)))
        bond_types.append(bond.GetBondType())

    fragmented = Chem.FragmentOnBonds(mol, bond_idx, dummyLabels=dummy_labels)
    mapping: list[tuple[int, ...]] = []
    frags = Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mapping)

    # global atom index -> (fragment id, index within that fragment)
    where = {a: (fid, i)
             for fid, atoms in enumerate(mapping)
             for i, a in enumerate(atoms)}

    # FragmentOnBonds appends the two dummies for broken bond k at
    # n + 2k and n + 2k + 1, in the order the bonds were passed.
    n = mol.GetNumAtoms()
    junctions = []
    for k, ((a1, a2), _) in enumerate(bonds):
        d1, d2 = where[n + 2 * k], where[n + 2 * k + 1]
        frag_a, frag_b = where[a1][0], where[a2][0]
        side_a = d1 if d1[0] == frag_a else d2
        side_b = d2 if d2[0] == frag_b else d1
        junctions.append((frag_a, side_a[1], frag_b, side_b[1], bond_types[k]))

    return list(frags), junctions


def reassemble(frags: list[Chem.Mol], junctions: list[tuple]) -> Chem.Mol:
    """Inverse of `decompose`. Rebuilds the parent, dummies removed."""
    combined, offsets = frags[0], [0]
    for frag in frags[1:]:
        offsets.append(combined.GetNumAtoms())
        combined = Chem.CombineMols(combined, frag)

    rw = Chem.RWMol(combined)
    doomed = []
    for frag_a, dummy_a, frag_b, dummy_b, bond_type in junctions:
        ga, gb = offsets[frag_a] + dummy_a, offsets[frag_b] + dummy_b
        # A dummy caps exactly one heavy atom; that atom is the real endpoint.
        anchor_a = rw.GetAtomWithIdx(ga).GetNeighbors()[0].GetIdx()
        anchor_b = rw.GetAtomWithIdx(gb).GetNeighbors()[0].GetIdx()
        rw.AddBond(anchor_a, anchor_b, bond_type)
        doomed += [ga, gb]

    # Descending, so earlier removals cannot shift later indices.
    for idx in sorted(doomed, reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def motif_graph(mol: Chem.Mol) -> tuple[list[str], list[tuple[int, int]]]:
    """Fragment SMILES and the undirected edges between them.

    This is the classifier's view: nodes are fragments, edges are the bonds
    BRICS cut. Because BRICS only cuts acyclic bonds, the motif graph of a
    connected molecule is a tree.
    """
    frags, junctions = decompose(mol)
    nodes = [Chem.MolToSmiles(f) for f in frags]
    edges = [(a, b) for a, _, b, _, _ in junctions]
    return nodes, edges


def atom_fragments(mol: Chem.Mol) -> tuple[list[int], list[tuple[int, int]]]:
    """Per-atom fragment id, plus the motif edges between fragments.

    The hierarchical classifier needs to know which fragment each ATOM belongs
    to -- `decompose` returns fragments as separate molecules, which loses the
    correspondence to the parent's atom indices. Returns `(frag_id, edges)`
    where `frag_id[i]` is the fragment holding atom `i` of `mol`, and `edges`
    are undirected `(frag_a, frag_b)` pairs.

    A zero-cut molecule gives `([0] * n_atoms, [])`: one fragment, no edges.
    That is ~1 molecule in 10 and needs no special case downstream.
    """
    bonds = list(BRICS.FindBRICSBonds(mol))
    n = mol.GetNumAtoms()
    if not bonds:
        return [0] * n, []

    bond_idx, dummy_labels = [], []
    for (a1, a2), (t1, t2) in bonds:
        bond_idx.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
        dummy_labels.append((int(t1), int(t2)))

    fragmented = Chem.FragmentOnBonds(mol, bond_idx, dummyLabels=dummy_labels)
    mapping: list[tuple[int, ...]] = []
    Chem.GetMolFrags(fragmented, asMols=True, fragsMolAtomMapping=mapping)

    frag_id = [-1] * n
    for fid, atoms in enumerate(mapping):
        for a in atoms:
            if a < n:                      # indices >= n are the added dummies
                frag_id[a] = fid
    assert -1 not in frag_id, "every atom must belong to exactly one fragment"

    edges = [(frag_id[a1], frag_id[a2]) for (a1, a2), _ in bonds]
    return frag_id, edges


def fragment_counts(smiles_list: list[str]) -> Counter:
    """How often each fragment occurs across a corpus. Unparseable input skipped."""
    counts: Counter = Counter()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        counts.update(Chem.MolToSmiles(f) for f in decompose(mol)[0])
    return counts


def vocab(smiles_list: list[str], min_count: int = 3) -> dict[str, int]:
    """Fragment -> contiguous id, with `UNK` always at 0.

    CALLERS MUST PASS THEIR OWN CORPUS, and the two halves pass different ones
    on purpose:

    * The **classifier** must build this from TRAINING-FOLD molecules only, per
      (dataset, seed). A vocabulary derived from the full dataset is
      transductive leakage under a scaffold split -- the split exists to give
      test molecules unseen cores, and a global vocabulary hands every test
      fragment its own embedding slot. Everything below the floor maps to `UNK`.

    * The **generator** may use the whole corpus (it makes no held-out claim)
      but needs a LOWER floor or none: frequency pruning that is a harmless long
      tail for a classifier is a reconstruction failure for a generator, which
      cannot emit a fragment it has no token for.

    Measured at `min_count=3`: BBBP keeps 316 fragments covering 75.1% of
    occurrences, B3DB keeps 1178 covering 81.3%. The often-quoted "~150
    fragments covers 95% of drug space" does not hold on these datasets.
    """
    counts = fragment_counts(smiles_list)
    keep = sorted(frag for frag, c in counts.items() if c >= min_count)
    return {UNK: 0, **{frag: i for i, frag in enumerate(keep, start=1)}}


def _self_check() -> None:
    aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    frags, junctions = decompose(aspirin)
    assert len(frags) == 4 and len(junctions) == 3, (len(frags), len(junctions))
    assert Chem.MolToSmiles(reassemble(frags, junctions)) == Chem.MolToSmiles(aspirin)
    nodes, edges = motif_graph(aspirin)
    assert len(nodes) == 4 and len(edges) == 3
    print(f"aspirin      -> {len(nodes)} motifs, {len(edges)} junctions: {nodes}")

    # Atoms must partition exactly: every heavy atom lands in one fragment.
    heavy = sum(sum(1 for a in f.GetAtoms() if a.GetAtomicNum() > 0) for f in frags)
    assert heavy == aspirin.GetNumAtoms(), (heavy, aspirin.GetNumAtoms())

    # The zero-cut fallback. Pyrene has no cleavable linker at all.
    pyrene = Chem.MolFromSmiles("c1ccc2c(c1)ccc1ccccc12")
    frags, junctions = decompose(pyrene)
    assert len(frags) == 1 and junctions == [], "pyrene must yield no BRICS cuts"
    assert Chem.MolToSmiles(reassemble(frags, junctions)) == Chem.MolToSmiles(pyrene)
    print("pyrene       -> 1 motif, 0 junctions (scaffold-core fallback)")

    # A double bond at a junction must survive. Rebuilding every junction as
    # SINGLE silently hydrogenates 91 BBBP molecules; this is that regression.
    alkene = Chem.MolFromSmiles("CN(C)CCC=C1c2ccccc2CCc2ccccc21")
    frags, junctions = decompose(alkene)
    back = Chem.MolToSmiles(reassemble(frags, junctions))
    assert back == Chem.MolToSmiles(alkene), f"bond order lost at junction: {back}"
    print("alkene       -> double bond preserved across the junction")

    # Chirality must survive decomposition, so the generator can draw from a
    # defined-stereo fragment pool even though the classifier cannot reward it.
    chiral = Chem.MolFromSmiles("CCN1CCC[C@H]1CNC(=O)c1cc(S(N)(=O)=O)ccc1OC")
    frags, junctions = decompose(chiral)
    assert Chem.MolToSmiles(reassemble(frags, junctions)) == Chem.MolToSmiles(chiral)
    assert any("@" in Chem.MolToSmiles(f) for f in frags), "stereocenter lost"
    print("sulpiride    -> stereocenter preserved in its fragment")

    # Single atoms appear in both datasets and must not crash.
    methane = Chem.MolFromSmiles("C")
    assert decompose(methane) == ([methane], []) or len(decompose(methane)[0]) == 1

    # atom_fragments must partition the atoms and agree with decompose.
    for smi in ("CC(=O)Oc1ccccc1C(=O)O", "c1ccc2c(c1)ccc1ccccc12", "C",
                "CN(C)CCC=C1c2ccccc2CCc2ccccc21"):
        m = Chem.MolFromSmiles(smi)
        fid, edges = atom_fragments(m)
        n_frags = len(set(fid))
        assert len(fid) == m.GetNumAtoms()
        assert n_frags == len(decompose(m)[0]), smi
        assert len(edges) == len(decompose(m)[1]), smi
        assert all(a != b for a, b in edges), "a motif edge must join two fragments"
        # BRICS only cuts acyclic bonds, so a connected molecule's motif graph
        # is a tree: edges == nodes - 1.
        assert len(edges) == n_frags - 1, f"{smi}: motif graph is not a tree"
    print("atom_fragments -> partitions atoms, motif graph is a tree")

    # Vocabulary: the floor is honoured and UNK is reserved at 0.
    corpus = ["CC(=O)Oc1ccccc1C(=O)O"] * 3 + ["c1ccccc1C(=O)O"]
    v = vocab(corpus, min_count=3)
    assert v[UNK] == 0
    assert all(i == j for j, i in enumerate(sorted(v.values())))
    rare = vocab(corpus, min_count=99)
    assert rare == {UNK: 0}, "nothing should clear a floor of 99"
    print(f"vocab        -> {len(v) - 1} fragments at min_count=3, UNK reserved at 0")

    print("\nself-check passed")


if __name__ == "__main__":
    _self_check()
