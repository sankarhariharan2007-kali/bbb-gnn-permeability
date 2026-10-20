"""Pick a representative set of generated survivors and render them."""
import sys, pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, Crippen, Descriptors
RDLogger.DisableLog("rdApp.*")

OUT = sys.argv[1]
d = pd.read_csv("results/candidates/brics10/screened.csv")
s = d[d.passes & ~d.in_training_flat].copy()
s["logp"] = [Crippen.MolLogP(Chem.MolFromSmiles(x)) for x in s.smiles]
s["tpsa"] = [Descriptors.TPSA(Chem.MolFromSmiles(x)) for x in s.smiles]
print(f"survivors: {len(s)}")

rng = np.random.default_rng(0)
groups = {
    "top by C (what the ranking picks)": s.nlargest(6, "c_score"),
    "random survivors (the actual population)": s.sample(6, random_state=0),
    # balanced: drug-like AND predicted permeable, which is what a shortlist
    # should actually be ranked on
    "balanced: QED > 0.8 and C > 0.95": s[(s.qed > 0.8) & (s.c_score > 0.95)].nlargest(6, "qed"),
}
for name, g in groups.items():
    print(f"\n=== {name}  (n={len(g)})")
    print(f"  mean C={g.c_score.mean():.3f}  QED={g.qed.mean():.3f}  "
          f"logP={g.logp.mean():.2f}  TPSA={g.tpsa.mean():.1f}  MW={g.mw.mean():.0f}")
    for _, r in g.iterrows():
        print(f"    C={r.c_score:.3f} QED={r.qed:.2f} logP={r.logp:+.2f} "
              f"TPSA={r.tpsa:5.1f} MW={r.mw:6.1f}  {r.smiles}")

print(f"\npopulation: C={s.c_score.mean():.3f} QED={s.qed.mean():.3f} "
      f"logP={s.logp.mean():.2f} TPSA={s.tpsa.mean():.1f} MW={s.mw.mean():.0f}")

rows = []
for name, g in groups.items():
    for _, r in g.iterrows():
        rows.append((r.smiles, f"C={r.c_score:.2f} QED={r.qed:.2f}\nlogP={r.logp:+.1f} MW={r.mw:.0f}"))
mols = [Chem.MolFromSmiles(x) for x, _ in rows]
img = Draw.MolsToGridImage(mols, molsPerRow=6, subImgSize=(300, 260),
                           legends=[l for _, l in rows], returnPNG=False)
img.save(OUT)
print(f"\nwrote {OUT}")
