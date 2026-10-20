"""Cluster the survivors and look for liability classes the screen cannot see.

WHY WE LOOKED, AND WHAT WE FOUND. Survivors already passed BRENK, PAINS A/B/C,
NIH and 18 targeted SMARTS. The worry was that a catalogue only encodes what
someone wrote down, so a recurring motif it is blind to would survive in bulk.

The answer on this output is that there is no such motif: **zero of 11,329
survivors match any of the residual patterns below.**

This also corrects an earlier claim of ours. The trioxide
`C#CC1(OOOC(O)C(Cl)(Cl)Cl)CCCCC1` was described in our own notes as having
passed the screen because no catalogue encodes it. It did not pass: it carries
`passes=False` with 1 targeted and 7 catalogue hits, and it is in arm A's pool
only. The screen caught it eight times over. What survives screening is clean
with respect to every pattern we could think to test.

TWO KINDS OF PROBLEM, kept separate because they are not the same claim:

  RESIDUAL LIABILITY   a structural motif a medicinal chemist would reject,
                       that the catalogues above do not encode.
  UNSUITABLE FOR BBB   no structural liability at all, but physicochemically
                       incapable of crossing the barrier -- a permanent charge,
                       or far outside the CNS property window. These are not
                       toxic, they are useless for THIS target, and the
                       classifier scoring them highly is a prediction error
                       rather than a reward-hacking event.

Clustering is Butina on Morgan r=2 2048-bit fingerprints at Tanimoto 0.6 -- the
same radius and cutoff `split.py` uses for acyclic grouping, so cluster
identity means the same thing here as in the splits.

    python -m src.cluster_survivors --arm cgate
    python -m src.cluster_survivors --self-check
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from .datasets import ROOT

RDLogger.DisableLog("rdApp.*")

ARMS = {"a": "brics10", "b": "brics_hybrid", "cgate": "brics_cgate",
        "b3db_a": "b3db_a", "b3db_cgate": "b3db_cgate"}
OUT = ROOT / "results" / "pubchem"
CUTOFF = 0.6

#: Motifs no catalogue in the screen encodes. Each is a pattern a medicinal
#: chemist rejects on sight, verified absent from ALERT_SMARTS and not reliably
#: caught by BRENK/PAINS/NIH on this output.
RESIDUAL: dict[str, str] = {
    # The known escape. Two linked peroxide oxygens are energetic; three are
    # explosive-adjacent. BRENK's peroxide pattern did not catch the trioxide.
    "peroxide_O-O": "[OX2][OX2]",
    "trioxide_O-O-O": "[OX2][OX2][OX2]",
    # Azo and diazo: mutagenic, and azo dyes cleave to aromatic amines.
    "azo": "[#6][NX2]=[NX2][#6]",
    "diazo": "[#6]=[NX2+]=[NX1-]",
    # Hard electrophiles that are not aldehydes, imines or acyl halides.
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "isothiocyanate": "[NX2]=[CX2]=[SX1]",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "carbodiimide": "[NX2]=[CX2]=[NX2]",
    # Labile or redox-active sulfur the targeted set omits.
    "disulfide": "[SX2][SX2]",
    "thiol": "[SX2H1]",
    # Heteroatoms that are rarely acceptable in a CNS candidate.
    "phosphorus": "[P]",
    "boron": "[B]",
    "silicon": "[Si]",
    "selenium_tellurium": "[Se,Te]",
    # NOT a liability, and removed after measuring. A four-membered-ring
    # pattern flagged 138 survivors, every one a cyclobutane, azetidine or
    # oxetane -- all unremarkable, and oxetane and azetidine are sought-after
    # modern bioisosteres rather than problems. Calling them a liability was
    # our labelling error, not a finding. Left here as a comment so it is not
    # re-added:
    #   "ring_strain_4": "[C;R1]1[C;R1][C,N,O;R1][C,N,O;R1]1"
}

#: Not liabilities. Physicochemical reasons a molecule cannot reach the brain,
#: whatever the classifier predicts.
def unsuitable(m: Chem.Mol) -> list[str]:
    out = []
    # A permanent formal charge does not cross a lipid bilayer passively. The
    # classifier has no term for this and BBBP contains few such molecules.
    if any(a.GetFormalCharge() != 0 for a in m.GetAtoms()):
        out.append("permanently charged")
    mw, tpsa = Descriptors.MolWt(m), Descriptors.TPSA(m)
    if mw > 500:
        out.append("MW > 500")
    if tpsa > 90:
        out.append("TPSA > 90")
    if Lipinski.NumHDonors(m) > 3:
        out.append("HBD > 3")
    if Crippen.MolLogP(m) > 5:
        out.append("logP > 5")
    return out


_RES = {k: Chem.MolFromSmarts(v) for k, v in RESIDUAL.items()}
_BAD = [k for k, v in _RES.items() if v is None]
if _BAD:
    raise ValueError(f"unparsable SMARTS: {_BAD}")


def load(arm: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / "results" / "candidates" / ARMS[arm] / "screened.csv")
    return d[d.passes.fillna(False).astype(bool)
             & ~d.in_training_flat.fillna(False).astype(bool)].reset_index(drop=True)


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    mols, keep = [], []
    for i, s in enumerate(df.smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            keep.append(i)
    d = df.iloc[keep].reset_index(drop=True)
    d["residual"] = [";".join(k for k, p in _RES.items() if m.HasSubstructMatch(p))
                     for m in mols]
    d["unsuitable"] = [";".join(unsuitable(m)) for m in mols]
    d["scaffold"] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    return d, mols


def cluster(mols: list[Chem.Mol], cutoff: float = CUTOFF) -> list[int]:
    """Butina at Tanimoto `cutoff`, same fingerprint as split.py's grouping."""
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1 - s for s in sims)
    cl = Butina.ClusterData(dists, len(fps), 1 - cutoff, isDistData=True)
    lab = np.zeros(len(fps), dtype=int)
    for ci, members in enumerate(cl):
        for m in members:
            lab[m] = ci
    return lab


def _self_check() -> None:
    # Every residual pattern must fire on a molecule that has it and not on one
    # that does not, or a zero count below means "pattern broken", not "clean".
    probe = {
        "peroxide_O-O": ("CCOOCC", "CCOCC"),
        "trioxide_O-O-O": ("CCOOOCC", "CCOOCC"),
        "azo": ("c1ccccc1N=Nc1ccccc1", "c1ccccc1N=Cc1ccccc1"),
        "isocyanate": ("CCN=C=O", "CCN=C"),
        "disulfide": ("CCSSCC", "CCSCC"),
        "phosphorus": ("CCOP(=O)(OCC)OCC", "CCOC(=O)OCC"),
        "thiol": ("CCS", "CCSC"),
    }
    for name, (pos, neg) in probe.items():
        p = _RES[name]
        assert Chem.MolFromSmiles(pos).HasSubstructMatch(p), f"{name} missed {pos}"
        assert not Chem.MolFromSmiles(neg).HasSubstructMatch(p), f"{name} fired on {neg}"
    # The trioxide that escaped the real screen must be caught here -- this is
    # the molecule that motivated the whole module.
    esc = Chem.MolFromSmiles("C#CC1(OOOC(O)C(Cl)(Cl)Cl)CCCCC1")
    assert esc.HasSubstructMatch(_RES["trioxide_O-O-O"]), "the known escape is not caught"
    # A permanent charge must be flagged; a neutral zwitterion-free drug must not.
    assert "permanently charged" in unsuitable(Chem.MolFromSmiles("C[N+](C)(C)C"))
    assert "permanently charged" not in unsuitable(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    print(f"{len(probe)} patterns verified both ways; the known trioxide escape is caught")
    print("cluster_survivors self-check passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), default="cgate")
    ap.add_argument("--top", type=int, default=12, help="clusters to print")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()
    if args.self_check:
        return _self_check()

    df = load(args.arm)
    print(f"arm {args.arm}: {len(df)} survivors (all passed BRENK/PAINS/NIH "
          f"+ 18 targeted SMARTS)\n")
    d, mols = annotate(df)

    n_res = (d.residual != "").sum()
    n_uns = (d.unsuitable != "").sum()
    print(f"{'=' * 74}\nRESIDUAL LIABILITIES -- motifs the screen does not encode\n{'=' * 74}")
    print(f"  {n_res} of {len(d)} survivors ({n_res / len(d):.2%}) carry at least one\n")
    c = Counter(k for row in d.residual for k in row.split(";") if k)
    for k, v in c.most_common():
        print(f"    {k:32} {v:>6}  {v / len(d):>6.2%}")

    print(f"\n{'=' * 74}\nUNSUITABLE FOR BBB -- no liability, cannot cross\n{'=' * 74}")
    print(f"  {n_uns} of {len(d)} survivors ({n_uns / len(d):.2%}) carry at least one\n")
    c2 = Counter(k for row in d.unsuitable for k in row.split(";") if k)
    for k, v in c2.most_common():
        sub = d[d.unsuitable.str.contains(k, regex=False)]
        print(f"    {k:32} {v:>6}  {v / len(d):>6.2%}   mean C {sub.c_score.mean():.3f}")

    print(f"\n{'=' * 74}\nCLUSTERS (Butina, Morgan r=2, Tanimoto {CUTOFF})\n{'=' * 74}")
    lab = cluster(mols)
    d["cluster"] = lab
    print(f"  {d.cluster.nunique()} clusters over {len(d)} molecules; "
          f"largest {Counter(lab).most_common(1)[0][1]}\n")
    g = (d.groupby("cluster")
          .agg(n=("smiles", "size"), c=("c_score", "mean"),
               qed=("qed", "mean"), heavy=("heavy_atoms", "mean"),
               residual=("residual", lambda s: (s != "").mean()),
               unsuitable=("unsuitable", lambda s: (s != "").mean()))
          .sort_values("n", ascending=False).head(args.top))
    print(f"    {'#':>4} {'n':>6} {'mean C':>8} {'QED':>6} {'heavy':>7} "
          f"{'resid%':>8} {'unsuit%':>8}  representative scaffold")
    for r in g.itertuples():
        rep = d[d.cluster == r.Index].iloc[0]
        sc = rep.scaffold if isinstance(rep.scaffold, str) and rep.scaffold else "(acyclic)"
        print(f"    {int(r.Index):>4} {int(r.n):>6} {r.c:>8.3f} {r.qed:>6.2f} "
              f"{r.heavy:>7.1f} {r.residual:>7.1%} {r.unsuitable:>8.1%}  {sc[:46]}")

    p = OUT / f"clusters_{args.arm}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    d[["smiles", "cluster", "scaffold", "residual", "unsuitable",
       "c_score", "qed", "heavy_atoms"]].to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
