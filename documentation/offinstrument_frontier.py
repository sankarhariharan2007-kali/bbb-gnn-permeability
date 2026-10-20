"""Instrument gap on the frontier points, reusing aggregate_arms' own pool()."""
import numpy as np, pandas as pd, torch
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from src.aggregate_arms import pool          # reused, not reimplemented
from src.datasets import ROOT, _load_raw
from src.reward import load_frozen_classifier
from src.train_gan import Discriminator

KL = [0.1, 0.3, 1.0, 3.0]
POINTS = ([("BRICS", k, Path(f"results/brics_sweep/kl{k}_ent0.0")) for k in KL] +
          [("SELFIES", k, Path(f"results/selfies_frontier/kl{k}")) for k in KL])

# reward_seed=0 is what every frontier run was trained against; b3db is the
# judge none of them ever saw.
bbbp = load_frozen_classifier(dataset="bbbp", seed=0)
b3db = load_frozen_classifier(dataset="b3db", models=("gin", "gat", "gine"), seed=0)
# The committed reference discriminator is BatchNorm-era: its state_dict has
# running_mean/running_var where the current GraphNorm default expects
# mean_scale. aggregate_arms.offinstrument() is therefore broken on this branch.
# Build it at the norm it was TRAINED with, which is also what run_contract does
# for the classifiers -- replay the recorded architecture, do not impose today's
# default. eval() keeps it a fixed judge either way.
disc = Discriminator(norm="batch")
disc.load_state_dict(torch.load(ROOT / "results/gan/seed0/bbbp_disc.pt", map_location="cpu"))
disc.eval()   # one fixed judge for every arm, as aggregate_arms does

real = [s for s in _load_raw("bbbp").query("label==1")["smiles"].dropna()
        if Chem.MolFromSmiles(s)]
rm = [Chem.MolFromSmiles(s) for s in real]
rb, r3 = float(bbbp(rm).mean()), float(b3db(rm).mean())

def w(p, c, lo=225, hi=250):
    return pd.read_csv(p / "history.csv")[c].iloc[lo:hi].mean()

rows = []
for fam, k, d in POINTS:
    smis = pool(d)
    mols = [Chem.MolFromSmiles(x) for x in smis]
    cb, c3 = float(bbbp(mols).mean()), float(b3db(mols).mean())
    rows.append(dict(fam=fam, kl=k, n=len(smis), scaffold=w(d, "scaffold_frac"),
                     c_log=w(d, "c_mean"), C_bbbp=cb, C_b3db=c3, gap=cb - c3,
                     D_real=float(disc.score(smis).mean())))
t = pd.DataFrame(rows)

print(f"REAL BBB+ drugs: C(BBBP) {rb:.3f}  C(B3DB) {r3:.3f}  gap {rb-r3:+.3f}"
      f"  D_real {float(disc.score(real).mean()):.3f}")
print("(the docstring's -0.029 was measured under the retired ensemble; this is")
print(" the same quantity recomputed under the current GraphNorm ensembles)\n")
t["gap_vs_real"] = (t.gap - (rb - r3)).abs()
print(t[["fam","kl","n","scaffold","C_bbbp","C_b3db","gap","gap_vs_real","D_real"]]
      .round(4).to_string(index=False))

print("\n=== the question: is BRICS@3.0's low C honesty or weakness? ===")
print("compare each arm's gap to the real-drug reference; SMALLER |gap-real| is")
print("more honest, i.e. less of the selector's blind spot being exploited.\n")
for fam in ("BRICS", "SELFIES"):
    s = t[t.fam == fam]
    print(f"  {fam:8s} " + "  ".join(f"kl{r.kl}: gap {r.gap:+.3f} (d{r.gap_vs_real:.3f})"
                                     for r in s.itertuples()))
print("\n  most honest overall:", end=" ")
best = t.loc[t.gap_vs_real.idxmin()]
print(f"{best.fam}@kl{best.kl} (gap {best.gap:+.3f}, |d| {best.gap_vs_real:.3f}, "
      f"scaffold {best.scaffold:.3f}, C_log {best.c_log:.3f})")
