"""Fraction of screened survivors whose Bemis-Murcko scaffold also occurs in the BBBP training corpus.
Stereochemistry is removed on both sides (generated molecules are flat); survivors without a ring
(empty scaffold) are excluded from the denominator.  usage: python paper/scaffold_overlap.py [repo]  (default: repository root)"""
import sys, pandas as pd
from rdkit import Chem, RDLogger; RDLogger.DisableLog("rdApp.*")
from rdkit.Chem.Scaffolds import MurckoScaffold as M
import pathlib
repo = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).resolve().parent.parent)
def scaf(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    try:
        Chem.RemoveStereochemistry(m); sc = M.GetScaffoldForMol(m); Chem.RemoveStereochemistry(sc)
        return Chem.MolToSmiles(sc)
    except Exception: return None
train = {scaf(s) for s in pd.read_csv(f"{repo}/data/raw/BBBP.csv").smiles} - {None, ""}
for name, p in [("two-run set (old)", "candidates/brics"), ("arm A, 10 seeds", "candidates/brics10"), ("gate arm, 10 seeds", "candidates/brics_cgate")]:
    d = pd.read_csv(f"{repo}/results/{p}/screened.csv"); sv = d[d.passes & ~d.in_training_flat]
    sc = [x for x in map(scaf, sv.smiles) if x not in (None, "")]
    print(f"{name:22s} survivors={len(sv):6d} cyclic={len(sc):6d} share={sum(x in train for x in sc)/len(sc):.3f}")
